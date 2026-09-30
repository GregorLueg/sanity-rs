//! The per-gene driver: sweep the variance grid, then aggregate over it.
//!
//! ### Memory
//!
//! SI eq. 42 wants the spread of `d*_c` about its own posterior mean, which
//! naively means holding `d*` and `var(d_c)` for every bin and every cell. It is
//! avoided: the offset `z_b` is one scalar per bin, so the first pass stores
//! `O(B)` state, and the second pass re-derives the per-cell quantities at a
//! known offset with a single sweep and no iteration. Scratch is `O(B + C)`.
//! The point-estimate rules skip the second pass entirely.
//!
//! ### Empty cells
//!
//! Neither pass evaluates the cells with no counts one by one. The first pass
//! reads their sums from [`ShiftTable`]; the second pass, and the point
//! estimates, evaluate them at the nodes of one Chebyshev fit in `ln T` and
//! interpolate, since an empty cell's outputs depend on it only through its
//! total. What is swept per bin is the gene's nonzero cells plus the nodes.
//!
//! Running the second pass cell block outermost instead, so that a block stays
//! in L1 across the whole grid, was measured on 2026-09-13 and was 10% slower:
//! the pass is bound by the transcendentals in the Wright omega and error bar
//! solves, not by memory traffic, so the tiling bought nothing and the extra
//! indexing cost.

use super::empty_cells::ShiftTable;
use super::fractions::{
    NonzeroCells, Stationary, SweepState, refresh, solve_stationary, solve_stationary_sparse,
};
use super::likelihood::{laplace, laplace_sparse};
use super::variance::cell_variance;
use crate::config::{SanityParams, VarianceGrid, VarianceRule};
use crate::errors::SanityErrors;
use crate::utils::chebyshev::Chebyshev;
use crate::utils::polygamma::{digamma, trigamma};

////////////
// Consts //
////////////

/// Posterior weight below which a bin is dropped from the second pass.
///
/// SI eq. 39 and 42 both weight a bin's contribution by `W_b`, so a bin under
/// this contributes less than this fraction of either estimate, against error
/// bars the method reports at the first or second digit. Every surviving bin
/// costs a full sweep over the cells plus an error bar solve in each of them,
/// which is the whole cost of the rule.
///
/// Measured 2026-09-13 on an M1 Max, 161 bins. On 400 simulated genes over
/// 4000 cells at 7.5% density the rule takes 6.78 s with nothing dropped,
/// 5.19 s at `1e-14`, 5.00 s here and 4.59 s at `1e-6`. Worst drift in the log
/// fold change, in units of the error bar the method reports for it, measured
/// on 200 genes over 2000 cells against an unpruned run: `4e-10` error bars at
/// `1e-12`, `1e-8` here, `1e-4` at `1e-6`. Almost all of the saving is already
/// had at `1e-14`, so this sits well inside the flat part of the curve.
///
/// Must stay strictly positive. It is also what stops the leading run of
/// underflowed bins from reaching the Welford update, where the first one would
/// divide a zero weight by a zero running sum.
pub(crate) const MARGINALISE_MIN_WEIGHT: f64 = 1e-10;

/// Fraction of cells with counts above which a gene sweeps densely.
///
/// The sparse sweep solves two roots per nonzero cell against one per cell for
/// the dense sweep. The empty-argument root sits at small `x` and converges in
/// fewer iterations, which is why the crossover lands at one half rather than
/// below it.
///
/// Measured 2026-09-30 on an M1 Max, `PosteriorMean`, 400 simulated genes over
/// 4000 cells, two interleaved passes. At 40.6% overall density: 1.93 s at
/// `0.4`, 1.87 s here, 1.87 s at `0.6`, 1.93 s at `0.7`, 2.09 s at `0.3`, 2.36 s
/// with no gate. At 59.8%: 2.45 s, 2.41 s, 2.40 s, 2.46 s, 2.57 s and 3.36 s.
const SPARSE_MAX_DENSITY: f64 = 0.5;

/// Chebyshev degree of the fit that fills in the empty cells.
///
/// An empty cell's outputs depend on the cell only through `ln T_c`, so the
/// second pass evaluates them at the nodes of one fit over the run's range of
/// `ln T` and interpolates to every empty cell.
///
/// Measured 2026-09-30 on an M1 Max, `Marginalise`, 400 simulated genes over
/// 4000 cells, worst log fold change against the per-cell pass in units of its
/// error bar. Library log sd 0.5: `1.7e-6` at degree 16, `3.5e-9` at 24,
/// `1.1e-11` at 32 and 48. Log sd 1.5, a spread well past droplet data:
/// `2.1e-5`, `1.2e-7`, `9.1e-10`, `5.9e-12`. Wall clock did not move with the
/// degree (0.49 to 0.53 s at sd 0.5), so this takes the degree that stays at
/// the per-cell pass's own noise across both.
const EMPTY_FIT_DEGREE: usize = 48;

/// Smallest width of the fit's interval in `ln T`.
///
/// Stops a run whose cells all have one total from giving a zero-width
/// interval; the nodes then simply extend past the data.
const EMPTY_FIT_MIN_WIDTH: f64 = 1.0;

/////////////////
// GeneScratch //
/////////////////

/// Per-thread scratch for one gene, reused across genes.
///
/// Allocated once per Rayon worker through `for_each_init`, never per gene.
#[derive(Clone, Debug)]
pub(crate) struct GeneScratch {
    /// Dense UMI counts for the gene in hand, length `n_cells`.
    counts: Vec<f64>,
    /// `omega(x_c)` at the current variance, length `n_cells`.
    omega: Vec<f64>,
    /// `ln omega(x_c)` at the current variance, length `n_cells`.
    log_omega: Vec<f64>,
    /// Running weighted mean of `d*_c` over bins, length `n_cells`.
    mean_d: Vec<f64>,
    /// Running weighted sum of squared deviations of `d*_c`, length `n_cells`.
    m2_d: Vec<f64>,
    /// Running weighted mean of `var(d_c)` over bins, length `n_cells`.
    mean_var: Vec<f64>,
    /// `ln P(k | v_b)` per bin, length `n_bins`.
    log_lik: Vec<f64>,
    /// `z(v_b)` per bin, length `n_bins`.
    offsets: Vec<f64>,
    /// `S_A(v_b)` per bin, length `n_bins`.
    curvature: Vec<f64>,
    /// Posterior weights `W_b`, length `n_bins`.
    weights: Vec<f64>,
    /// The gene's nonzero cells, for the sparse first pass.
    nonzero: NonzeroCells,
    /// Counts of the cells the second pass evaluates, see [`load_cells`].
    cell_counts: Vec<f64>,
    /// `ln T` of the cells the second pass evaluates.
    cell_log_totals: Vec<f64>,
    /// Node values of the empty-cell fit, `d` then `e`.
    fit_values: Vec<f64>,
    /// Chebyshev coefficients of the empty-cell fit, `d` then `e`.
    fit_coeffs: Vec<f64>,
    /// What `omega` and `log_omega` currently hold, for warm starting. In the
    /// second pass they hold the compact cells of [`load_cells`] instead.
    ///
    /// Cleared at the start of every gene: the arrays survive across genes but
    /// their contents belong to the gene that wrote them.
    state: SweepState,
}

impl GeneScratch {
    /// Allocate scratch for a run.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells.
    /// * `n_bins` - Number of variance bins.
    ///
    /// ### Returns
    ///
    /// Zeroed scratch of `O(n_cells + n_bins)`.
    pub(crate) fn new(n_cells: usize, n_bins: usize) -> Self {
        Self {
            counts: vec![0.0; n_cells],
            omega: vec![0.0; n_cells],
            log_omega: vec![0.0; n_cells],
            mean_d: vec![0.0; n_cells],
            m2_d: vec![0.0; n_cells],
            mean_var: vec![0.0; n_cells],
            log_lik: vec![0.0; n_bins],
            offsets: vec![0.0; n_bins],
            curvature: vec![0.0; n_bins],
            weights: vec![0.0; n_bins],
            nonzero: NonzeroCells::new(n_cells),
            cell_counts: Vec::with_capacity(n_cells),
            cell_log_totals: Vec::with_capacity(n_cells),
            fit_values: Vec::new(),
            fit_coeffs: Vec::new(),
            state: None,
        }
    }
}

//////////////
// Frontend //
//////////////

/////////////
// Helpers //
/////////////

/// First pass: solve the offset and the marginal likelihood in every bin.
///
/// The grid ascends in `v` and `z` grows with `v`, so each bin warm starts the
/// Newton solve from its predecessor's offset.
///
/// ### Params
///
/// * `s` - `K`, the total UMI count of this gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `log_total_sum` - `ln(sum_c T_c)`.
/// * `grid` - The variance grid.
/// * `table` - The run's empty-cell sums, or `None` to sweep every cell. With
///   it, only `scratch.nonzero` is swept and the dense `omega` scratch is left
///   untouched.
/// * `scratch` - Per-thread scratch; `log_lik`, `offsets` and `curvature` are
///   filled.
///
/// ### Returns
///
/// Nothing, or a solver failure.
fn sweep_grid(
    s: f64,
    log_totals: &[f64],
    log_total_sum: f64,
    grid: &VarianceGrid,
    table: Option<&ShiftTable>,
    scratch: &mut GeneScratch,
) -> Result<(), SanityErrors> {
    let n_cells = log_totals.len();
    let mut guess = log_total_sum + 0.5 * grid.values[0];
    for (b, &v) in grid.values.iter().enumerate() {
        let (point, fit) = match table {
            Some(table) => {
                let point = solve_stationary_sparse(
                    v,
                    s,
                    n_cells,
                    guess,
                    table,
                    log_totals,
                    &mut scratch.nonzero,
                )?;
                let fit = laplace_sparse(&point, n_cells, table, log_totals, &scratch.nonzero);
                (point, fit)
            }
            None => {
                let point = solve_stationary(
                    v,
                    s,
                    &scratch.counts,
                    log_totals,
                    guess,
                    &mut scratch.state,
                    &mut scratch.omega,
                    &mut scratch.log_omega,
                )?;
                let fit = laplace(
                    &point,
                    &scratch.counts,
                    log_totals,
                    &scratch.omega,
                    &scratch.log_omega,
                );
                (point, fit)
            }
        };
        scratch.log_lik[b] = fit.log_marginal;
        scratch.offsets[b] = point.z;
        scratch.curvature[b] = point.curvature_sum;
        guess = point.z;
    }
    Ok(())
}

/// Normalise the bin log-likelihoods into posterior weights.
///
/// SI eq. 34. The scale prior on `v` is already carried by the grid being
/// uniform in `ln v`, so this is a plain softmax, shifted by the maximum.
///
/// ### Params
///
/// * `log_lik` - `ln P(k | v_b)` per bin.
/// * `weights` - Output, `W_b` per bin, summing to one.
///
/// ### Returns
///
/// Nothing; `weights` is overwritten, or
/// [`SanityErrors::NonFiniteBinLikelihood`] if a bin is unusable. The check is
/// up front because `fold(NEG_INFINITY, f64::max)` skips a NaN, which would
/// then propagate silently into every output for the gene.
pub(crate) fn posterior_weights(log_lik: &[f64], weights: &mut [f64]) -> Result<(), SanityErrors> {
    if let Some((bin, &value)) = log_lik.iter().enumerate().find(|(_, l)| !l.is_finite()) {
        return Err(SanityErrors::NonFiniteBinLikelihood { bin, value });
    }
    let peak = log_lik.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for (w, &l) in weights.iter_mut().zip(log_lik) {
        *w = (l - peak).exp();
        total += *w;
    }
    let scale = if total > 0.0 { 1.0 / total } else { 0.0 };
    for w in weights.iter_mut() {
        *w *= scale;
    }
    Ok(())
}

/// Second pass: integrate the per-cell estimates over the variance posterior.
///
/// SI eq. 39 and 42. `z_b` is already known, so each bin costs one sweep with no
/// iteration. The spread of `d*_c` across bins accumulates by weighted Welford,
/// which is SI eq. 42 rather than the algebraically equivalent SI eq. 41 and so
/// does not lose digits when the log fold change is large. Only the cells of
/// [`load_cells`] are swept.
///
/// ### Params
///
/// * `s` - `K`, the total UMI count of this gene.
/// * `indices` - Cell indices of this gene's stored counts.
/// * `log_totals` - `ln T_c` for every cell.
/// * `grid` - The variance grid.
/// * `fit` - The run's empty-cell fit.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
///
/// ### Returns
///
/// The gene-level summary.
#[allow(clippy::too_many_arguments)]
fn marginalise(
    s: f64,
    indices: &[u32],
    log_totals: &[f64],
    grid: &VarianceGrid,
    fit: &Chebyshev,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) -> GeneSummary {
    // The gene-level scalars do not touch the cells, so they come out of the
    // cell loop entirely and are accumulated once over the grid.
    let (summary, weight_sum) =
        marginal_summary(s, &grid.values, &scratch.weights, &scratch.offsets);

    let interpolate = load_cells(indices, log_totals, fit, scratch);
    let mut running = 0.0;
    for (b, &v) in grid.values.iter().enumerate() {
        let weight = scratch.weights[b];
        if weight < MARGINALISE_MIN_WEIGHT {
            continue;
        }
        let point = Stationary {
            v,
            s,
            log_vs: (v * s).ln(),
            z: scratch.offsets[b],
            curvature_sum: scratch.curvature[b],
        };
        running += weight;
        accumulate_bin(&point, weight, running, scratch);
    }
    write_cells(
        indices,
        log_totals,
        fit,
        interpolate,
        weight_sum,
        scratch,
        out_fold_change,
        out_error,
    );

    summary
}

/// Load the cells the second pass evaluates into the compact arrays.
///
/// An empty cell's `d*_c` and `var(d_c)` depend on the cell only through
/// `ln T_c`, at every bin. With more empty cells than `fit` has nodes, the
/// compact arrays hold the gene's nonzero cells, in `indices` order, followed by
/// one stand-in empty cell at each node. Otherwise they hold every cell in order
/// and nothing is interpolated. Clears the Welford accumulators and the sweep
/// state, since the arrays change meaning.
///
/// ### Params
///
/// * `indices` - Cell indices of this gene's stored counts.
/// * `log_totals` - `ln T_c` for every cell.
/// * `fit` - The run's empty-cell fit.
/// * `scratch` - Per-thread scratch, with `counts` scattered.
///
/// ### Returns
///
/// Whether the empty cells are to be interpolated.
fn load_cells(
    indices: &[u32],
    log_totals: &[f64],
    fit: &Chebyshev,
    scratch: &mut GeneScratch,
) -> bool {
    let n_cells = log_totals.len();
    let n_nonzero = indices
        .iter()
        .filter(|&&i| scratch.counts[i as usize] > 0.0)
        .count();
    let interpolate = n_cells - n_nonzero > fit.n_points();

    scratch.cell_counts.clear();
    scratch.cell_log_totals.clear();
    if interpolate {
        for &i in indices {
            let k = scratch.counts[i as usize];
            if k > 0.0 {
                scratch.cell_counts.push(k);
                scratch.cell_log_totals.push(log_totals[i as usize]);
            }
        }
        for j in 0..fit.n_points() {
            scratch.cell_counts.push(0.0);
            scratch.cell_log_totals.push(fit.node(j));
        }
    } else {
        scratch
            .cell_counts
            .extend_from_slice(&scratch.counts[..n_cells]);
        scratch.cell_log_totals.extend_from_slice(log_totals);
    }

    let n = scratch.cell_counts.len();
    scratch.mean_d[..n].fill(0.0);
    scratch.m2_d[..n].fill(0.0);
    scratch.mean_var[..n].fill(0.0);
    scratch.state = None;
    interpolate
}

/// Add one bin to the Welford accumulators of the compact cells.
///
/// ### Params
///
/// * `point` - The stationary point of this bin.
/// * `weight` - The bin's posterior weight.
/// * `running` - Total weight so far, this bin included.
/// * `scratch` - Per-thread scratch, loaded by [`load_cells`].
fn accumulate_bin(point: &Stationary, weight: f64, running: f64, scratch: &mut GeneScratch) {
    let n = scratch.cell_counts.len();
    refresh(
        point,
        &scratch.cell_counts,
        &scratch.cell_log_totals,
        &mut scratch.state,
        &mut scratch.omega[..n],
        &mut scratch.log_omega[..n],
    );

    let share = weight / running;
    for c in 0..n {
        let d = point.log_fold_change(scratch.log_omega[c], scratch.cell_log_totals[c]);
        let var = cell_variance(point, scratch.cell_counts[c], d, scratch.omega[c]);

        let delta = d - scratch.mean_d[c];
        scratch.mean_d[c] += share * delta;
        scratch.m2_d[c] += weight * delta * (d - scratch.mean_d[c]);
        scratch.mean_var[c] += share * (var - scratch.mean_var[c]);
    }
}

/// Write `d_c` and `e_c` for every cell from the compact accumulators.
///
/// Nonzero cells are written from their own accumulators; empty cells, when
/// interpolated, from the fit through the node values.
///
/// ### Params
///
/// * `indices` - Cell indices of this gene's stored counts.
/// * `log_totals` - `ln T_c` for every cell.
/// * `fit` - The run's empty-cell fit.
/// * `interpolate` - What [`load_cells`] returned.
/// * `weight_sum` - Total weight of the accumulated bins.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
#[allow(clippy::too_many_arguments)]
fn write_cells(
    indices: &[u32],
    log_totals: &[f64],
    fit: &Chebyshev,
    interpolate: bool,
    weight_sum: f64,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) {
    let error = |scratch: &GeneScratch, c: usize| {
        (scratch.mean_var[c] + scratch.m2_d[c] / weight_sum).sqrt()
    };

    if !interpolate {
        for c in 0..log_totals.len() {
            out_fold_change[c] = scratch.mean_d[c];
            out_error[c] = error(scratch, c);
        }
        return;
    }

    let p = fit.n_points();
    let first_node = scratch.cell_counts.len() - p;
    scratch.fit_values.resize(2 * p, 0.0);
    scratch.fit_coeffs.resize(2 * p, 0.0);
    for j in 0..p {
        scratch.fit_values[j] = scratch.mean_d[first_node + j];
        scratch.fit_values[p + j] = error(scratch, first_node + j);
    }
    let (values_d, values_e) = scratch.fit_values.split_at(p);
    let (coeffs_d, coeffs_e) = scratch.fit_coeffs.split_at_mut(p);
    fit.coefficients(values_d, coeffs_d);
    fit.coefficients(values_e, coeffs_e);

    for (c, &lt) in log_totals.iter().enumerate() {
        out_fold_change[c] = fit.eval(coeffs_d, lt);
        out_error[c] = fit.eval(coeffs_e, lt);
    }
    let mut m = 0;
    for &i in indices {
        let i = i as usize;
        if scratch.counts[i] > 0.0 {
            out_fold_change[i] = scratch.mean_d[m];
            out_error[i] = error(scratch, m);
            m += 1;
        }
    }
}

/// The gene-level summary of the marginalising rule, from the per-bin offsets.
///
/// SI eq. 39 and 42 for `m`, its error bar and `<v>`, over the bins that
/// survive [`MARGINALISE_MIN_WEIGHT`]. Touches no cell.
///
/// ### Params
///
/// * `s` - `K`, the total UMI count of this gene.
/// * `grid` - The variance grid.
/// * `weights` - Posterior weights `W_b`.
/// * `offsets` - `z(v_b)` per bin.
///
/// ### Returns
///
/// The summary, and the total weight of the surviving bins.
pub(crate) fn marginal_summary(
    s: f64,
    grid: &[f64],
    weights: &[f64],
    offsets: &[f64],
) -> (GeneSummary, f64) {
    let mut weight_sum = 0.0;
    let mut mean_offset = 0.0;
    let mut m2_offset = 0.0;
    let mut mean_variance = 0.0;
    for ((&v, &weight), &z) in grid.iter().zip(weights).zip(offsets) {
        if weight < MARGINALISE_MIN_WEIGHT {
            continue;
        }
        weight_sum += weight;
        let share = weight / weight_sum;
        let delta = z - mean_offset;
        mean_offset += share * delta;
        m2_offset += weight * delta * (z - mean_offset);
        mean_variance += share * (v - mean_variance);
    }
    let summary = GeneSummary {
        mean_log_quotient: digamma(s) - mean_offset,
        mean_log_quotient_error: (trigamma(s) + m2_offset / weight_sum).sqrt(),
        variance: mean_variance,
    };
    (summary, weight_sum)
}

/// Where a collapsing rule evaluates, and the offset it warm starts from.
///
/// SPEC section 7. [`VarianceRule::MaxPosterior`] takes the most probable bin
/// and its offset; [`VarianceRule::PosteriorMean`] takes `<v>`, which lands
/// between bins, and the offset of the bin nearest it.
///
/// ### Params
///
/// * `rule` - The collapsing rule; anything but `MaxPosterior` is treated as
///   `PosteriorMean`.
/// * `grid` - The variance grid, ascending.
/// * `weights` - Posterior weights `W_b`.
/// * `offsets` - `z(v_b)` per bin.
///
/// ### Returns
///
/// The variance to evaluate at, the offset guess, and the posterior mean `<v>`.
pub(crate) fn collapse_target(
    rule: VarianceRule,
    grid: &[f64],
    weights: &[f64],
    offsets: &[f64],
) -> (f64, f64, f64) {
    let posterior_mean: f64 = grid.iter().zip(weights).map(|(v, w)| v * w).sum();

    let (v, guess) = match rule {
        VarianceRule::MaxPosterior => {
            let best = weights
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |acc, (b, &w)| {
                    if w > acc.1 { (b, w) } else { acc }
                })
                .0;
            (grid[best], offsets[best])
        }
        _ => {
            // The grid ascends, so the first bin at or above `<v>` and the one
            // below it bracket it; take the closer of the two.
            let above = grid.partition_point(|&v| v < posterior_mean);
            let nearest = match above {
                0 => 0,
                b if b == grid.len() => b - 1,
                b if posterior_mean - grid[b - 1] <= grid[b] - posterior_mean => b - 1,
                b => b,
            };
            (posterior_mean, offsets[nearest])
        }
    };
    (v, guess, posterior_mean)
}

/// Collapse the variance posterior to a single value and evaluate there.
///
/// SPEC section 7. [`VarianceRule::MaxPosterior`] reuses the offset stored for
/// that bin as its guess; [`VarianceRule::PosteriorMean`] lands between bins,
/// warm started from the offset of the bin nearest `<v>`. Either re-solves,
/// sparse when the first pass was.
///
/// ### Params
///
/// * `rule` - The collapsing rule.
/// * `s` - `K`, the total UMI count of this gene.
/// * `indices` - Cell indices of this gene's stored counts.
/// * `log_totals` - `ln T_c` for every cell.
/// * `grid` - The variance grid.
/// * `table` - The empty-cell sums, if the first pass was sparse.
/// * `fit` - The run's empty-cell fit.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
///
/// ### Returns
///
/// The gene-level summary, or a solver failure.
#[allow(clippy::too_many_arguments)]
fn collapse(
    rule: VarianceRule,
    s: f64,
    indices: &[u32],
    log_totals: &[f64],
    grid: &VarianceGrid,
    table: Option<&ShiftTable>,
    fit: &Chebyshev,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) -> Result<GeneSummary, SanityErrors> {
    let (v, guess, posterior_mean) =
        collapse_target(rule, &grid.values, &scratch.weights, &scratch.offsets);

    let point = match table {
        Some(table) => {
            // Cold per-cell state: the last sweep sat at the top of the grid,
            // too far from `v` for the warm start's prediction.
            scratch.nonzero.state = None;
            solve_stationary_sparse(
                v,
                s,
                log_totals.len(),
                guess,
                table,
                log_totals,
                &mut scratch.nonzero,
            )?
        }
        None => solve_stationary(
            v,
            s,
            &scratch.counts,
            log_totals,
            guess,
            &mut scratch.state,
            &mut scratch.omega,
            &mut scratch.log_omega,
        )?,
    };
    point_estimate(
        &point,
        indices,
        log_totals,
        fit,
        scratch,
        out_fold_change,
        out_error,
    );

    Ok(GeneSummary {
        mean_log_quotient: digamma(s) - point.z,
        mean_log_quotient_error: trigamma(s).sqrt(),
        variance: posterior_mean,
    })
}

/// Write `d_c` and `e_c` from a single stationary point.
///
/// Used by every rule that does not integrate over the grid, where the error bar
/// is the posterior width at one variance and nothing is added for the spread of
/// `d*_c` across variances. One bin of weight one through the second pass's
/// machinery, so the empty cells are interpolated the same way.
///
/// ### Params
///
/// * `point` - The stationary point.
/// * `indices` - Cell indices of this gene's stored counts.
/// * `log_totals` - `ln T_c` for every cell.
/// * `fit` - The run's empty-cell fit.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
fn point_estimate(
    point: &Stationary,
    indices: &[u32],
    log_totals: &[f64],
    fit: &Chebyshev,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) {
    let interpolate = load_cells(indices, log_totals, fit, scratch);
    accumulate_bin(point, 1.0, 1.0, scratch);
    write_cells(
        indices,
        log_totals,
        fit,
        interpolate,
        1.0,
        scratch,
        out_fold_change,
        out_error,
    );
}

/// The run's fit for the empty cells, over the range of `ln T`.
///
/// ### Params
///
/// * `log_totals` - `ln T_c` for every cell.
///
/// ### Returns
///
/// Degree [`EMPTY_FIT_DEGREE`] interpolation over the cells' `ln T`, at least
/// [`EMPTY_FIT_MIN_WIDTH`] wide.
pub(crate) fn empty_cell_fit(log_totals: &[f64]) -> Chebyshev {
    let lo = log_totals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = log_totals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Chebyshev::new(lo, hi.max(lo + EMPTY_FIT_MIN_WIDTH), EMPTY_FIT_DEGREE)
}

/// The log marginal likelihood of one gene at one variance, in `f64`.
///
/// One offset solve and one Laplace fit, SI eq. 27 and 33. The GPU path calls
/// this for the few bins its own `f32` likelihood cannot separate.
///
/// ### Params
///
/// * `v` - The variance.
/// * `s` - `K`, the total UMI count of this gene.
/// * `counts` - Dense UMI counts for this gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `guess` - Starting offset.
/// * `state` - What `omega` and `log_omega` hold, as for
///   [`solve_stationary`]; `None` on the first call for a gene, so that a run
///   of calls over ascending `v` warm starts each from the last.
/// * `omega` - Scratch, length `n_cells`.
/// * `log_omega` - Scratch, length `n_cells`.
///
/// ### Returns
///
/// `ln P(k | v)` and the offset `z(v)`, or a solver failure.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn log_marginal_at(
    v: f64,
    s: f64,
    counts: &[f64],
    log_totals: &[f64],
    guess: f64,
    state: &mut SweepState,
    omega: &mut [f64],
    log_omega: &mut [f64],
) -> Result<(f64, f64), SanityErrors> {
    let point = solve_stationary(v, s, counts, log_totals, guess, state, omega, log_omega)?;
    let fit = laplace(&point, counts, log_totals, omega, log_omega);
    Ok((fit.log_marginal, point.z))
}

/////////////////
// GeneSummary //
/////////////////

/// Everything one gene contributes to the run's output.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GeneSummary {
    /// Posterior mean of the gene's log transcription quotient, `m`.
    pub mean_log_quotient: f64,
    /// Error bar on `m`.
    pub mean_log_quotient_error: f64,
    /// Posterior estimate of the gene's variance in log fold change.
    pub variance: f64,
}

/// Run one gene end to end.
///
/// Writes `d_c` into `out_fold_change` and `e_c` into `out_error`, both of
/// length `n_cells`, and returns the gene-level summary.
///
/// ### Params
///
/// * `indices` - Cell indices of this gene's stored counts.
/// * `values` - The stored counts, aligned with `indices`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `log_total_sum` - `ln(sum_c T_c)`, the seed for the first offset solve.
/// * `grid` - The variance grid.
/// * `table` - The run's empty-cell sums, if built. Used for the first pass
///   when the gene is at most [`SPARSE_MAX_DENSITY`] dense.
/// * `fit` - The run's empty-cell fit, from [`empty_cell_fit`].
/// * `params` - Run parameters.
/// * `scratch` - Per-thread scratch.
/// * `out_fold_change` - Output, `d_c` for every cell.
/// * `out_error` - Output, `e_c` for every cell.
///
/// ### Returns
///
/// The gene-level summary, or a solver failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gene(
    indices: &[u32],
    values: &[u32],
    log_totals: &[f64],
    log_total_sum: f64,
    grid: &VarianceGrid,
    table: Option<&ShiftTable>,
    fit: &Chebyshev,
    params: &SanityParams,
    scratch: &mut GeneScratch,
    out_fold_change: &mut [f64],
    out_error: &mut [f64],
) -> Result<GeneSummary, SanityErrors> {
    let n_cells = log_totals.len();

    // Scatter the sparse column. Only the touched entries are cleared again at
    // the end, so this stays O(nnz) rather than O(n_cells) for the reset.
    scratch.state = None;
    let mut total_counts = 0.0;
    for (&i, &k) in indices.iter().zip(values) {
        scratch.counts[i as usize] = k as f64;
        total_counts += k as f64;
    }
    // SPEC section 1: the 1/alpha prior makes the exponent K, not K + 1.
    // `prepare_run` rejects K = 0, where the posterior is improper.
    let s = total_counts;

    let summary = match params.variance_rule {
        VarianceRule::Fixed(v) => {
            let point = solve_stationary(
                v,
                s,
                &scratch.counts,
                log_totals,
                log_total_sum + 0.5 * v,
                &mut scratch.state,
                &mut scratch.omega,
                &mut scratch.log_omega,
            )?;
            point_estimate(
                &point,
                indices,
                log_totals,
                fit,
                scratch,
                out_fold_change,
                out_error,
            );
            GeneSummary {
                mean_log_quotient: digamma(s) - point.z,
                mean_log_quotient_error: trigamma(s).sqrt(),
                variance: v,
            }
        }
        rule => {
            let table =
                table.filter(|_| indices.len() as f64 <= SPARSE_MAX_DENSITY * n_cells as f64);
            if table.is_some() {
                scratch.nonzero.load(indices, values, log_totals);
            }
            sweep_grid(s, log_totals, log_total_sum, grid, table, scratch)?;
            posterior_weights(&scratch.log_lik, &mut scratch.weights)?;
            match rule {
                VarianceRule::Marginalise => marginalise(
                    s,
                    indices,
                    log_totals,
                    grid,
                    fit,
                    scratch,
                    out_fold_change,
                    out_error,
                ),
                _ => collapse(
                    rule,
                    s,
                    indices,
                    log_totals,
                    grid,
                    table,
                    fit,
                    scratch,
                    out_fold_change,
                    out_error,
                )?,
            }
        }
    };

    for &i in indices {
        scratch.counts[i as usize] = 0.0;
    }

    Ok(summary)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Verbosity;
    use crate::simulate::{SimulationParams, simulate};

    #[test]
    fn test_empty_cell_fit_matches_per_cell_pass() {
        let sim = simulate(Some(SimulationParams {
            n_genes: 30,
            n_cells: 400,
            library_size: 100.0,
            library_log_sd: 1.0,
            ..SimulationParams::default()
        }))
        .expect("valid simulation");
        let n_cells = sim.cell_totals.len();
        let log_totals: Vec<f64> = sim.cell_totals.iter().map(|t| t.ln()).collect();
        let log_total_sum = sim.cell_totals.iter().sum::<f64>().ln();
        let grid = VarianceGrid::new(1e-3, 50.0, 60);
        let table = ShiftTable::new(&log_totals, 50.0 * 1e5);

        let fit = empty_cell_fit(&log_totals);
        // More nodes than cells, so `load_cells` sweeps every cell directly.
        let exact = Chebyshev::new(0.0, 1.0, n_cells);

        for rule in [
            VarianceRule::Marginalise,
            VarianceRule::PosteriorMean,
            VarianceRule::Fixed(1.0),
        ] {
            let params = SanityParams::new(rule, 1e-3, 50.0, 60, Verbosity::Quiet);
            let mut scratch = GeneScratch::new(n_cells, grid.len());
            let (mut d_fit, mut e_fit) = (vec![0.0; n_cells], vec![0.0; n_cells]);
            let (mut d_all, mut e_all) = (vec![0.0; n_cells], vec![0.0; n_cells]);
            for g in 0..sim.counts.n_genes() {
                let (indices, values) = sim.counts.gene(g);
                if values.is_empty() {
                    continue;
                }
                for (f, d, e) in [
                    (&fit, &mut d_fit, &mut e_fit),
                    (&exact, &mut d_all, &mut e_all),
                ] {
                    run_gene(
                        indices,
                        values,
                        &log_totals,
                        log_total_sum,
                        &grid,
                        Some(&table),
                        f,
                        &params,
                        &mut scratch,
                        d,
                        e,
                    )
                    .expect("converges");
                }
                for c in 0..n_cells {
                    let err = (d_fit[c] - d_all[c]).abs() / e_all[c];
                    assert!(err < 1e-9, "{rule:?} gene {g} cell {c}: {err:e} error bars");
                    let err = (e_fit[c] - e_all[c]).abs() / e_all[c];
                    assert!(
                        err < 1e-9,
                        "{rule:?} gene {g} cell {c}: error bar off by {err:e}"
                    );
                }
            }
        }
    }
}
