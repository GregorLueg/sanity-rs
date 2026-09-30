//! The stationary point of the log posterior at a fixed variance.
//!
//! Everything here is per gene and dense in the cell axis: a cell with no
//! counts for this gene still contributes through its library size. The sparse
//! sweep keeps that: it reads the empty cells' share from
//! [`super::empty_cells::ShiftTable`] and sweeps only the nonzero cells.
//!
//! The stationarity condition is one Wright omega evaluation per cell plus a
//! single scalar root find for the normalisation offset `z`. Writing
//! `u_c = v s w_c` turns SI eq. 25 into `u_c + ln u_c = x_c`, and the constraint
//! `sum_c w_c = 1` becomes `sum_c omega(x_c) = v s`, which is strictly
//! decreasing in `z` and so has a unique root.

use super::empty_cells::{EmptySums, ShiftTable};
use crate::errors::SanityErrors;
use crate::utils::wright_omega::{log_omega, log_omega_near, omega_from_log};

/// Relative tolerance on the offset residual `sum_c omega_c - v s`.
///
/// Measured against `v s` rather than absolutely, because `v s` spans many
/// orders of magnitude across genes and variance bins.
const OFFSET_TOL: f64 = 1e-13;

/// Ulp budget for the naive summation behind the offset residual.
///
/// `sum_c omega_c` is accumulated in one pass over `n_cells` terms, so its own
/// rounding noise grows as `C * eps` and overtakes [`OFFSET_TOL`] well before
/// `1e5` cells. Measured 2026-09-13: one gene, counts 1-7 in every third cell,
/// uniform `T_c = 5000`, 21 bins, failed at `C = 20_000` with residual
/// `2.68e-11` against `v s = 26 681`. Four ulp per term covers the worst
/// ordering of that sum.
const OFFSET_SUM_ULPS: f64 = 4.0;

/// Bracket width at which the offset is as resolved as `f64` allows.
///
/// Scaled by `1 + |z|` because `z` is a logarithm and can sit either side of
/// one. Once the bracket is this narrow, bisection cannot separate the two ends
/// and the residual test can only be met by luck.
const OFFSET_BRACKET_ULPS: f64 = 4.0;

/// Iteration cap for the Newton solve on the offset.
///
/// Newton on a monotone function with an exact derivative, bracketed and warm
/// started from the neighbouring variance bin, settles in a handful of steps.
/// The cap turns a pathological input into an error rather than a hang.
const OFFSET_MAX_ITER: usize = 100;

/// Cap on the bracket expansion for the offset.
///
/// Each step doubles, so this covers a span of `2^60` around the initial guess,
/// far wider than any real data can require.
const OFFSET_MAX_BRACKET: usize = 60;

/// First outward step when bracketing the offset from a warm start.
///
/// The Newton loop starts from whichever end of the bracket flipped the sign,
/// so this is also how far the first Newton iterate can sit from the guess.
/// Sweeping an ordered grid moves `z` by far less than one unit between bins,
/// so a small first step usually flips on the first try and leaves a bracket
/// narrow enough for Newton to close in two iterations.
///
/// Measured 2026-09-13 on an M1 Max, 200 simulated genes over 4000 cells at
/// 7.5% density and 161 bins, two interleaved passes. Wall clock for
/// `PosteriorMean`: 1.0 gives 2.65 s, 0.25 gives 1.97 s, 0.125 gives 1.85 s,
/// 0.0625 gives 1.53 s, 0.03125 gives 1.49 s, 0.015625 gives 1.46 s. The curve
/// is flat below 0.03125, so anything smaller only buys extra doublings when a
/// guess turns out worse than this data makes it.
const OFFSET_BRACKET_STEP_WARM: f64 = 0.03125;

/// First outward step when bracketing the offset from a cold start.
///
/// A cold guess can be a unit or more from the root, and every doubling costs a
/// full sweep over the cells, so starting as small as the warm step just buys
/// four extra sweeps before the bracket is wide enough to contain the root.
///
/// Measured 2026-09-13, same machine and data: sharing the warm step's value
/// put `VarianceRule::Fixed`, which is one cold solve per gene and nothing
/// else, at 0.105 s against 0.051 s here.
const OFFSET_BRACKET_STEP_COLD: f64 = 1.0;

/// The stationary point of one gene at one variance.
///
/// Holds only what downstream kernels need. `omega` and `log_omega` are indexed
/// by cell; the rest are scalars for the gene.
#[derive(Debug)]
pub(crate) struct Stationary {
    /// The variance this point was solved at.
    pub v: f64,
    /// `K`, the total UMI count of the gene.
    pub s: f64,
    /// `ln(v s)`, cached because every per-cell quantity needs it.
    pub log_vs: f64,
    /// The normalisation offset `z`, with `exp(z) = sum_c T_c exp(d*_c)`.
    pub z: f64,
    /// `S_A = sum_c omega_c / (1 + omega_c)` at this point.
    ///
    /// Carried on the point rather than recomputed because the offset solve's
    /// Newton derivative is the same reduction:
    /// `F'(z) = -sum_c omega_c / (1 + omega_c)`. The solve's last sweep already
    /// accumulated it at the returned `z`, so both the Laplace fit and the
    /// variance kernel read it from here instead of making another pass.
    pub curvature_sum: f64,
}

impl Stationary {
    /// The log fold change in one cell.
    ///
    /// SI eq. 28. `d*_c = ln w_c - ln T_c + z` and `ln w_c = t_c - ln(v s)`.
    ///
    /// ### Params
    ///
    /// * `log_omega` - `ln omega(x_c)` for this cell.
    /// * `log_total` - `ln T_c` for this cell.
    ///
    /// ### Returns
    ///
    /// `d*_c` at this variance.
    #[inline(always)]
    pub(crate) fn log_fold_change(&self, log_omega: f64, log_total: f64) -> f64 {
        log_omega - self.log_vs - log_total + self.z
    }

    /// The normalised weight in one cell.
    ///
    /// `w_c = omega_c / (v s)`, summing to one over cells by construction.
    ///
    /// ### Params
    ///
    /// * `omega` - `omega(x_c)` for this cell.
    ///
    /// ### Returns
    ///
    /// `w_c` at this variance.
    #[inline(always)]
    pub(crate) fn weight(&self, omega: f64) -> f64 {
        omega / (self.v * self.s)
    }
}

/// The two reductions one sweep over the cells produces.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sweep {
    /// `sum_c omega(x_c)`, which the offset solve drives to `v s`.
    pub sum_omega: f64,
    /// `sum_c omega_c / (1 + omega_c)`. This is `-F'(z)` for the offset solve
    /// and `S_A` for everything downstream; one pass produces both.
    pub curvature_sum: f64,
}

/// What argument the `omega` and `log_omega` scratch currently holds.
///
/// `x_c = v k_c + ln T_c + shift`, so the pair `(v, shift)` pins the whole
/// vector and lets the next sweep compute its own `dx` per cell.
pub(crate) type SweepState = Option<(f64, f64)>;

/// Fill `omega` and `log_omega` for every cell at a given offset.
///
/// `x_c = v k_c + ln T_c + ln(v s) - z`. When `state` says what the scratch
/// already holds, each cell warm starts from its own previous root through
/// [`log_omega_near`] rather than from the cold guess. The argument moves by
/// `dx_c = (v - v_prev) k_c + (shift - shift_prev)`, which is exact, so the
/// only thing a stale state costs is iterations, never correctness.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `log_vs` - `ln(v s)`.
/// * `z` - The current offset.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `state` - What the scratch holds; updated to this sweep on return.
/// * `out_omega` - Output, `omega(x_c)`.
/// * `out_log_omega` - Output, `ln omega(x_c)`.
///
/// ### Returns
///
/// Both reductions over the cells.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    v: f64,
    log_vs: f64,
    z: f64,
    counts: &[f64],
    log_totals: &[f64],
    state: &mut SweepState,
    out_omega: &mut [f64],
    out_log_omega: &mut [f64],
) -> Sweep {
    let shift = log_vs - z;
    let (v_prev, shift_prev) = state.unwrap_or((0.0, 0.0));
    let warm = state.is_some();
    let dv = v - v_prev;
    let dshift = shift - shift_prev;

    let mut sum_omega = 0.0;
    let mut curvature_sum = 0.0;

    for (((&k, &lt), w), t) in counts
        .iter()
        .zip(log_totals)
        .zip(out_omega.iter_mut())
        .zip(out_log_omega.iter_mut())
    {
        let x = v * k + lt + shift;
        *t = if warm {
            log_omega_near(x, dv * k + dshift, *t, *w)
        } else {
            log_omega(x)
        };
        *w = omega_from_log(x, *t);
        sum_omega += *w;
        curvature_sum += *w / (1.0 + *w);
    }

    *state = Some((v, shift));

    Sweep {
        sum_omega,
        curvature_sum,
    }
}

//////////////////
// NonzeroCells //
//////////////////

/// A gene's nonzero cells, compacted, with their per-cell sweep state.
///
/// The sparse sweep needs two roots per cell: at the cell's own argument, and
/// at the argument it would have with no counts, which is its term inside the
/// table's all-cell sum and has to come back out. Both warm start like the
/// dense sweep.
#[derive(Clone, Debug)]
pub(crate) struct NonzeroCells {
    /// `k_c`.
    counts: Vec<f64>,
    /// `ln T_c`.
    log_totals: Vec<f64>,
    /// `omega(v k_c + ln T_c + shift)`.
    omega: Vec<f64>,
    /// `ln` of `omega`.
    log_omega: Vec<f64>,
    /// `omega(ln T_c + shift)`, the cell's term in the table.
    omega_empty: Vec<f64>,
    /// `ln` of `omega_empty`.
    log_omega_empty: Vec<f64>,
    /// What the four arrays hold, as for the dense [`SweepState`].
    pub state: SweepState,
}

impl NonzeroCells {
    /// Allocate for at most `n_cells` nonzero cells.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells.
    ///
    /// ### Returns
    ///
    /// Empty scratch with capacity for every cell.
    pub(crate) fn new(n_cells: usize) -> Self {
        Self {
            counts: Vec::with_capacity(n_cells),
            log_totals: Vec::with_capacity(n_cells),
            omega: Vec::with_capacity(n_cells),
            log_omega: Vec::with_capacity(n_cells),
            omega_empty: Vec::with_capacity(n_cells),
            log_omega_empty: Vec::with_capacity(n_cells),
            state: None,
        }
    }

    /// Load one gene's nonzero cells and clear the sweep state.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices of the stored counts.
    /// * `values` - The stored counts.
    /// * `log_totals` - `ln T_c` for every cell.
    pub(crate) fn load(&mut self, indices: &[u32], values: &[u32], log_totals: &[f64]) {
        self.counts.clear();
        self.log_totals.clear();
        for (&i, &k) in indices.iter().zip(values) {
            if k > 0 {
                self.counts.push(k as f64);
                self.log_totals.push(log_totals[i as usize]);
            }
        }
        let n = self.counts.len();
        for a in [
            &mut self.omega,
            &mut self.log_omega,
            &mut self.omega_empty,
            &mut self.log_omega_empty,
        ] {
            a.clear();
            a.resize(n, 0.0);
        }
        self.state = None;
    }

    /// One sweep, with the empty cells' share taken from the all-cell sums.
    ///
    /// The sums over the gene's empty cells are `all` minus the nonzero cells'
    /// empty-argument terms, so each cell adds its own term and removes that
    /// one.
    ///
    /// ### Params
    ///
    /// * `v` - The variance bin.
    /// * `shift` - `ln(v s) - z`.
    /// * `all` - The table's sums over every cell at `shift`.
    ///
    /// ### Returns
    ///
    /// Both reductions over all cells of the gene.
    fn sweep(&mut self, v: f64, shift: f64, all: EmptySums) -> Sweep {
        let (v_prev, shift_prev) = self.state.unwrap_or((0.0, 0.0));
        let warm = self.state.is_some();
        let dv = v - v_prev;
        let dshift = shift - shift_prev;

        let mut sum_omega = 0.0;
        let mut curvature_sum = 0.0;
        for c in 0..self.counts.len() {
            let k = self.counts[c];
            let lt = self.log_totals[c];

            let x = v * k + lt + shift;
            let t = if warm {
                log_omega_near(x, dv * k + dshift, self.log_omega[c], self.omega[c])
            } else {
                log_omega(x)
            };
            let w = omega_from_log(x, t);

            let x0 = lt + shift;
            let t0 = if warm {
                log_omega_near(x0, dshift, self.log_omega_empty[c], self.omega_empty[c])
            } else {
                log_omega(x0)
            };
            let w0 = omega_from_log(x0, t0);

            self.log_omega[c] = t;
            self.omega[c] = w;
            self.log_omega_empty[c] = t0;
            self.omega_empty[c] = w0;
            sum_omega += w - w0;
            curvature_sum += w / (1.0 + w) - w0 / (1.0 + w0);
        }
        self.state = Some((v, shift));

        Sweep {
            sum_omega: all.omega + sum_omega,
            curvature_sum: all.curvature + curvature_sum,
        }
    }

    /// The nonzero cells' share of the Laplace reductions.
    ///
    /// Each is the cell's own term minus its empty-argument term, so adding the
    /// table's all-cell sum gives the reduction over every cell. An empty
    /// cell's `d_c` is `-omega_c`, which is why `sum_c d_c^2` over the table is
    /// `sum_c omega_c^2`.
    ///
    /// ### Params
    ///
    /// * `point` - The stationary point the arrays were last swept at.
    ///
    /// ### Returns
    ///
    /// `(sum d^2 - omega_0^2, sum k d, sum ln(1 + omega) - ln(1 + omega_0))`.
    pub(crate) fn laplace_terms(&self, point: &Stationary) -> (f64, f64, f64) {
        let mut sum_sq = 0.0;
        let mut sum_data = 0.0;
        let mut sum_log_diag = 0.0;
        for c in 0..self.counts.len() {
            let d = point.log_fold_change(self.log_omega[c], self.log_totals[c]);
            let w0 = self.omega_empty[c];
            sum_sq += d * d - w0 * w0;
            sum_data += self.counts[c] * d;
            sum_log_diag += self.omega[c].ln_1p() - w0.ln_1p();
        }
        (sum_sq, sum_data, sum_log_diag)
    }
}

/// Solve the stationarity condition for one gene at one variance.
///
/// SI eq. 27, over the dense cell arrays. See [`solve_offset`] for the solve.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `s` - `K`, the total UMI count of this gene.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `guess` - Starting offset. Pass the previous bin's solution when sweeping
///   an ordered grid; otherwise `ln(sum_c T_c) + v / 2`.
/// * `state` - What the scratch holds, threaded through [`evaluate`].
/// * `omega` - Scratch and output, `omega(x_c)` at the solution.
/// * `log_omega` - Scratch and output, `ln omega(x_c)` at the solution.
///
/// ### Returns
///
/// The stationary point, or a solver failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_stationary(
    v: f64,
    s: f64,
    counts: &[f64],
    log_totals: &[f64],
    guess: f64,
    state: &mut SweepState,
    omega: &mut [f64],
    log_omega: &mut [f64],
) -> Result<Stationary, SanityErrors> {
    let cold = state.is_none();
    solve_offset(v, s, counts.len(), guess, cold, |log_vs, z| {
        evaluate(v, log_vs, z, counts, log_totals, state, omega, log_omega)
    })
}

/// Solve the stationarity condition, summing empty cells from the table.
///
/// SI eq. 27, with the reductions over cells with no counts taken from
/// [`ShiftTable`] and only the gene's nonzero cells swept. Leaves the dense
/// `omega` scratch untouched; a caller that needs per-cell values afterwards
/// refreshes it.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `s` - `K`, the total UMI count of this gene.
/// * `n_cells` - Number of cells, for the residual tolerance.
/// * `guess` - Starting offset.
/// * `table` - The run's table of empty-cell sums.
/// * `cells` - The gene's nonzero cells and their warm start state.
///
/// ### Returns
///
/// The stationary point, or a solver failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_stationary_sparse(
    v: f64,
    s: f64,
    n_cells: usize,
    guess: f64,
    table: &ShiftTable,
    cells: &mut NonzeroCells,
) -> Result<Stationary, SanityErrors> {
    let cold = cells.state.is_none();
    solve_offset(v, s, n_cells, guess, cold, |log_vs, z| {
        let shift = log_vs - z;
        let all = table.eval(shift);
        cells.sweep(v, shift, all)
    })
}

/// The offset solve, over whatever sweep the caller supplies.
///
/// SI eq. 27. Brackets the root of `F(z) = sum_c omega(x_c) - v s` by doubling,
/// then runs Newton with the exact derivative `F'(z) = -sum_c omega_c / (1 + omega_c)`,
/// falling back to bisection whenever a Newton step leaves the bracket.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `s` - `K`, the total UMI count of this gene.
/// * `n_cells` - Number of cells, for the residual tolerance.
/// * `guess` - Starting offset.
/// * `cold` - Whether the sweep has no previous state to warm start from.
/// * `sweep` - Takes `(ln(v s), z)` and returns both reductions at that `z`,
///   leaving its per-cell state consistent with it.
///
/// ### Returns
///
/// The stationary point, or [`SanityErrors::FractionSolveDiverged`] if neither
/// the bracket nor the Newton loop settles. A root whose bracket has collapsed
/// to a few ulp is returned regardless of the residual.
fn solve_offset(
    v: f64,
    s: f64,
    n_cells: usize,
    guess: f64,
    cold: bool,
    mut sweep: impl FnMut(f64, f64) -> Sweep,
) -> Result<Stationary, SanityErrors> {
    let vs = v * s;
    let log_vs = vs.ln();
    let tol = vs * OFFSET_TOL.max(OFFSET_SUM_ULPS * n_cells as f64 * f64::EPSILON);

    // Every sweep leaves its per-cell state, `f` and `fit.curvature_sum`
    // consistent with the `z` it was given, so no point is ever evaluated
    // twice: the bracket search hands its last sweep straight to Newton, and
    // the returned point's curvature is the one the final sweep accumulated.
    let mut z = guess;
    let mut fit = sweep(log_vs, z);
    let mut f = fit.sum_omega - vs;

    // `F` is strictly decreasing in `z`, so a positive residual means `z` is too
    // small. Expand outwards by doubling until the sign flips.
    let (mut lo, mut hi) = (z, z);
    let ascending = f > 0.0;
    let mut step = if cold {
        OFFSET_BRACKET_STEP_COLD
    } else {
        OFFSET_BRACKET_STEP_WARM
    };
    loop {
        if ascending {
            hi = lo + step;
            z = hi;
        } else {
            lo = hi - step;
            z = lo;
        }
        fit = sweep(log_vs, z);
        f = fit.sum_omega - vs;
        if (ascending && f <= 0.0) || (!ascending && f >= 0.0) {
            break;
        }
        if ascending {
            lo = hi
        } else {
            hi = lo
        }
        step *= 2.0;
        if step > (1u64 << OFFSET_MAX_BRACKET) as f64 {
            return Err(SanityErrors::FractionSolveDiverged {
                iterations: 0,
                residual: f,
            });
        }
    }

    for iteration in 0..OFFSET_MAX_ITER {
        if f.abs() <= tol {
            return Ok(Stationary {
                v,
                s,
                log_vs,
                z,
                curvature_sum: fit.curvature_sum,
            });
        }
        if f > 0.0 {
            lo = z
        } else {
            hi = z
        }
        // The root is converged to machine precision; the residual test above
        // its own summation noise can no longer be met.
        if hi - lo <= OFFSET_BRACKET_ULPS * f64::EPSILON * (1.0 + z.abs()) {
            return Ok(Stationary {
                v,
                s,
                log_vs,
                z,
                curvature_sum: fit.curvature_sum,
            });
        }

        // F'(z) = -sum_c omega_c / (1 + omega_c), which the sweep at `z`
        // already accumulated.
        let slope = fit.curvature_sum;
        let next = if slope > 0.0 { z + f / slope } else { f64::NAN };

        z = if next.is_finite() && next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };

        if iteration + 1 == OFFSET_MAX_ITER {
            return Err(SanityErrors::FractionSolveDiverged {
                iterations: OFFSET_MAX_ITER,
                residual: f,
            });
        }

        fit = sweep(log_vs, z);
        f = fit.sum_omega - vs;
    }

    // The loop above always returns.
    unreachable!()
}

/// Re-evaluate the per-cell state at an offset that is already known.
///
/// The second pass of the marginalising rule (SPEC section 6.1) has `z(v_b)`
/// from the first pass, so it needs no iteration at all: one sweep refills
/// `omega` and `log_omega`.
///
/// ### Params
///
/// * `point` - The stationary point from the first pass.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `state` - What the scratch holds, threaded through [`evaluate`].
/// * `omega` - Output, `omega(x_c)`.
/// * `log_omega` - Output, `ln omega(x_c)`.
///
/// ### Returns
///
/// Nothing; `omega` and `log_omega` are overwritten.
pub(crate) fn refresh(
    point: &Stationary,
    counts: &[f64],
    log_totals: &[f64],
    state: &mut SweepState,
    omega: &mut [f64],
    log_omega: &mut [f64],
) {
    evaluate(
        point.v,
        point.log_vs,
        point.z,
        counts,
        log_totals,
        state,
        omega,
        log_omega,
    );
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A small deterministic gene: counts, cell totals, and a variance.
    fn fixture() -> (Vec<f64>, Vec<f64>, f64, f64) {
        let counts = vec![0.0, 3.0, 1.0, 0.0, 7.0, 0.0, 2.0, 0.0];
        let totals = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0, 450.0, 1330.0, 1010.0];
        let s = counts.iter().sum::<f64>() + 1.0;
        (counts, totals, s, 0.75)
    }

    #[test]
    fn test_fractions_sum_to_one() {
        let (counts, totals, s, v) = fixture();
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = totals.iter().sum::<f64>().ln() + v / 2.0;

        let point = solve_stationary(
            v,
            s,
            &counts,
            &log_totals,
            guess,
            &mut None,
            &mut omega,
            &mut log_omega,
        )
        .expect("the offset solve converges on well formed input");

        let sum: f64 = omega.iter().map(|&w| point.weight(w)).sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn test_stationarity_condition_holds() {
        // The gradient of SI eq. 20 must vanish: -d_c / v + k_c - s w_c = 0.
        let (counts, totals, s, v) = fixture();
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = totals.iter().sum::<f64>().ln();

        let point = solve_stationary(
            v,
            s,
            &counts,
            &log_totals,
            guess,
            &mut None,
            &mut omega,
            &mut log_omega,
        )
        .expect("converges");

        for c in 0..counts.len() {
            let d = point.log_fold_change(log_omega[c], log_totals[c]);
            let grad = -d / v + counts[c] - s * point.weight(omega[c]);
            assert!(grad.abs() < 1e-9, "cell {c} gradient {grad:e}");
        }
    }

    #[test]
    fn test_offset_recovers_the_definition() {
        // exp(z) = sum_c T_c exp(d*_c).
        let (counts, totals, s, v) = fixture();
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = 0.0;

        let point = solve_stationary(
            v,
            s,
            &counts,
            &log_totals,
            guess,
            &mut None,
            &mut omega,
            &mut log_omega,
        )
        .expect("converges from a deliberately poor guess");

        let lhs: f64 = (0..counts.len())
            .map(|c| totals[c] * point.log_fold_change(log_omega[c], log_totals[c]).exp())
            .sum();
        assert_relative_eq!(lhs, point.z.exp(), max_relative = 1e-11);
    }

    #[test]
    fn test_small_variance_limit() {
        // As v -> 0 the equations collapse to exp(z) = sum_c T_c exp(v k_c).
        let (counts, totals, s, _) = fixture();
        let v = 1e-8;
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = totals.iter().sum::<f64>().ln();

        let point = solve_stationary(
            v,
            s,
            &counts,
            &log_totals,
            guess,
            &mut None,
            &mut omega,
            &mut log_omega,
        )
        .expect("converges");

        let expected: f64 = (0..counts.len())
            .map(|c| totals[c] * (v * counts[c]).exp())
            .sum::<f64>()
            .ln();
        // The limit is approached to first order in `v s`, since
        // `omega(x) = e^x (1 - e^x + ...)` and `e^x` is of order
        // `v s T_c / sum_c T_c` here. Anything tighter would be testing the
        // truncation of that series rather than the solver.
        assert_relative_eq!(point.z, expected, max_relative = 1e-7);
    }
}
