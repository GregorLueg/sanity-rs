//! The two device kernels and their per-cell helpers.
//!
//! [`fn@sweep_grid_gpu`] is the first pass: one workgroup of whole planes per
//! gene, running the whole variance grid with the offset solve inside the
//! workgroup. [`fn@marginalise_gpu`]
//! is the second pass: one thread per gene and cell, integrating over the bins
//! the host kept. Everything is `f32`; see the [`crate::gpu`] module doc for
//! where that costs precision and how the arithmetic is arranged around it.

// The `#[cube]` macro generates undocumented expansion items.
#![allow(missing_docs)]

use cubecl::prelude::*;

use crate::model::empty_cells::{N_SUMS, PANEL_POINTS};

////////////
// Consts //
////////////

/// Totals one sweep reduces; see `sweep`.
pub const N_TOTALS: u32 = 5;

/// Most planes one gene's workgroup may hold, which also sizes the shared
/// scratch for the cross-plane reduction: `N_TOTALS * MAX_PLANES_PER_GENE`
/// values, 320 bytes.
pub const MAX_PLANES_PER_GENE: u32 = 16;

/// Resolution of the anchored offset `zeta`, in ulp of `1 + |zeta|`.
///
/// The solve stops once the next Newton step, `F / S_A`, or the bracket is
/// narrower than this. Same role as the CPU's `OFFSET_BRACKET_ULPS`.
///
/// The device solves for `zeta = z - a_b` rather than `z` itself, with the
/// anchor `a_b = ln(sum_c T_c) + v_b / 2` held by the host in `f64`. `z` is of
/// order 20, where an `f32` ulp is `2e-6`, and the Laplace determinant is not
/// stationary in `z`: its slope, `0.5 sum_c omega_c / (1 + omega_c)^2`, runs to
/// `1e4` for an expressed gene over 200000 cells.
const OFFSET_ULPS_F32: f32 = 4.0;

/// Iteration cap on the Newton solve for the offset. Same as the CPU's cap: a
/// pathological input becomes a status code rather than a hang.
const OFFSET_MAX_ITER: u32 = 100;

/// Cap on the bracket doublings for the offset, a span of `2^60`.
const OFFSET_MAX_BRACKET: u32 = 60;

/// First bracket step from a warm start. Carried over from the CPU's
/// `OFFSET_BRACKET_STEP_WARM`; not re-measured on the device.
const OFFSET_BRACKET_STEP_WARM: f32 = 0.03125;

/// First bracket step from a cold start. Carried over from the CPU's
/// `OFFSET_BRACKET_STEP_COLD`; not re-measured on the device.
const OFFSET_BRACKET_STEP_COLD: f32 = 1.0;

/// Residual tolerance of the Wright omega refinement, in ulp of the largest
/// term of `exp(t) + t - x`.
///
/// Scaled by `1 + e^t + |t|` rather than the CPU's `1 + e^t`: in `f32` the
/// rounding of `t` itself dominates for `x` of a few units below zero, which is
/// where every empty cell sits, and the CPU's scaling would run every one of
/// them to the iteration cap.
const OMEGA_TOL_F32: f32 = 4.0 * f32::EPSILON;

/// Iteration cap on the Halley refinement. Every call starts cold, and Halley
/// converges cubically from the asymptotic guess.
const OMEGA_MAX_ITER: u32 = 8;

/// Above this argument the starting guess is `ln(x - ln x)`; below it, `x`.
/// Same threshold as the CPU's `OMEGA_LARGE_X`.
const OMEGA_LARGE_X: f32 = 1.0;

/// The half-unit drop in log posterior that defines the empty-cell error bar,
/// SI eq. 38.
const HALF_UNIT_DROP: f32 = 0.5;

/// Residual tolerance on the half-unit drop, in ulp of the drop.
const SIGMA_TOL_F32: f32 = 4.0 * f32::EPSILON;

/// Bracket width, in ulp of `sigma`, at which the half-unit drop is resolved.
const SIGMA_BRACKET_ULPS_F32: f32 = 4.0;

/// Iteration cap on the half-unit drop solve. Same as the CPU's.
const SIGMA_MAX_ITER: u32 = 60;

/// Cap on the bracket doublings for the half-unit drop. Same as the CPU's.
const SIGMA_MAX_BRACKET: u32 = 40;

/// Largest `|y|` for which `ln(1 + y)` goes through its series.
///
/// WGSL has no `log1p`, and the textbook `ln(u) * y / (u - 1)` correction
/// relies on `(1 + y) - 1` not folding to `y`, which Metal's fast-math is free
/// to do. The series in `u = y / (2 + y)` has no cancellation; at `y = 0.4`,
/// `u^2 = 0.028` and the first dropped term is `u^12 / 13 = 3.6e-11` relative,
/// and at `y = -0.4` it is `u^2 = 0.0625` and `1.9e-15`. Outside it,
/// `ln(1 + y)` is already accurate to a few ulp.
const LOG1P_SERIES_MAX: f32 = 0.4;

/// Largest argument `exp(x) - 1` takes through its Taylor series.
///
/// At `x = 0.5` the first dropped term, `x^10 / 10!`, is `2.7e-10` relative.
/// Above it `exp(x) >= 1.65`, so the subtraction loses at most two bits.
const EXPM1_SERIES_MAX: f32 = 0.5;

/////////////
// Helpers //
/////////////

/// `ln r` for `r = 1 + y`, from whichever of the two is exact.
///
/// Near one the series in `y` keeps full relative precision; away from it
/// `ln r` does. A caller that has `r` as an exact ratio passes it here rather
/// than forming `1 + y`, which loses `y`'s digits as `r` approaches zero.
///
/// ### Params
///
/// * `y` - `r - 1`, above minus one.
/// * `r` - `1 + y`, positive.
///
/// ### Returns
///
/// `ln r`.
#[cube]
fn ln_near_one<F: Float>(y: F, r: F) -> F {
    let one = F::new(1.0_f32);
    let mut out = F::ln(r);
    if F::abs(y) < F::new(LOG1P_SERIES_MAX) {
        let u = y / (F::new(2.0_f32) + y);
        let u2 = u * u;
        let tail = F::new(1.0_f32 / 11.0_f32) * u2 + F::new(1.0_f32 / 9.0_f32);
        let tail = tail * u2 + F::new(1.0_f32 / 7.0_f32);
        let tail = tail * u2 + F::new(1.0_f32 / 5.0_f32);
        let tail = tail * u2 + F::new(1.0_f32 / 3.0_f32);
        out = F::new(2.0) * u * (tail * u2 + one);
    }
    out
}

/// `ln(1 + y)` for `y > -1`, accurate for small `|y|`.
///
/// ### Params
///
/// * `y` - The argument, above minus one.
///
/// ### Returns
///
/// `ln(1 + y)`.
#[cube]
fn log1p<F: Float>(y: F) -> F {
    ln_near_one::<F>(y, F::new(1.0_f32) + y)
}

/// `exp(x) - 1` for `x >= 0`, accurate for small `x`.
///
/// ### Params
///
/// * `x` - The argument, non-negative.
///
/// ### Returns
///
/// `exp(x) - 1`.
#[cube]
fn expm1<F: Float>(x: F) -> F {
    let one = F::new(1.0_f32);
    let mut out = F::exp(x) - one;
    if x < F::new(EXPM1_SERIES_MAX) {
        let mut acc = one + x / F::new(9.0_f32);
        acc = one + x / F::new(8.0_f32) * acc;
        acc = one + x / F::new(7.0_f32) * acc;
        acc = one + x / F::new(6.0_f32) * acc;
        acc = one + x / F::new(5.0_f32) * acc;
        acc = one + x / F::new(4.0_f32) * acc;
        acc = one + x / F::new(3.0_f32) * acc;
        acc = one + x / F::new(2.0_f32) * acc;
        out = x * acc;
    }
    out
}

/// `ln omega(x)`: the solution `t` of `exp(t) + t = x`, from cold.
///
/// SI eq. 25-26 through the Wright omega function, as in
/// `crate::utils::wright_omega`. Halley, falling back to Newton wherever the
/// Halley denominator is not positive.
///
/// ### Params
///
/// * `x` - The argument.
///
/// ### Returns
///
/// `t = ln omega(x)`.
#[cube]
fn log_omega<F: Float>(x: F) -> F {
    let one = F::new(1.0_f32);
    let two = F::new(2.0_f32);
    let mut t = x;
    if x > F::new(OMEGA_LARGE_X) {
        t = F::ln(x - F::ln(x));
    }
    let mut i = 0u32;
    while i < OMEGA_MAX_ITER {
        let e = F::exp(t);
        let g = e + t - x;
        if F::abs(g) < F::new(OMEGA_TOL_F32) * (one + e + F::abs(t)) {
            break;
        }
        let d1 = e + one;
        let halley = two * d1 * d1 - g * e;
        let mut step = g / d1;
        if halley > F::new(0.0_f32) {
            step = two * g * d1 / halley;
        }
        t -= step;
        i += 1u32;
    }
    t
}

/// `omega(x)` from `x` and `t = ln omega(x)`, on the branch that does not
/// cancel.
///
/// ### Params
///
/// * `x` - The argument.
/// * `t` - `ln omega(x)`.
///
/// ### Returns
///
/// `omega(x)`.
#[cube]
fn omega_from_log<F: Float>(x: F, t: F) -> F {
    let mut w = x - t;
    if x <= F::new(0.0_f32) {
        w = F::exp(t);
    }
    w
}

/// The log fold change of one cell, SI eq. 28, in the form that does not cancel.
///
/// Two exact forms. The stationarity condition gives `d = v k - omega`, which
/// is accurate relative to `v k` wherever `omega` itself is (its exponential
/// branch, and every empty cell, where it is `-omega`). The general form
/// `t - ln T_c - (ln(v s) - z)` adds and subtracts terms of order ten, so its
/// absolute error is a few `1e-6`; that is kept only where `omega > 0.57` and
/// `v k` is correspondingly large. Downstream, `d / (v k)` must be accurate
/// for `ln(1 - d / (v k))`, which is what decides the split.
///
/// ### Params
///
/// * `k` - The cell's count.
/// * `x` - The Wright omega argument.
/// * `t` - `ln omega(x_c)`.
/// * `w` - `omega(x_c)`.
/// * `v` - The variance bin.
/// * `lt` - `ln T_c`.
/// * `shift` - `ln(v s) - z`.
///
/// ### Returns
///
/// `d*_c`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn log_fold_change<F: Float>(k: F, x: F, t: F, w: F, v: F, lt: F, shift: F) -> F {
    let zero = F::new(0.0_f32);
    let mut d = v * k - w;
    if k > zero && x > zero {
        d = t - lt - shift;
    }
    d
}

/// One sweep of a gene over its cells at a given offset, reduced over the
/// workgroup.
///
/// Each lane strides the cells by the workgroup width. The partials are summed
/// within each plane, then across planes through shared memory in plane order,
/// so every thread receives bit-identical totals and takes the same branch in
/// the offset solve; the barriers therefore sit in uniform control flow.
///
/// Spreading a gene over several planes is as much for precision as for
/// occupancy: each lane's partial is a sequential `f32` sum, and its rounding
/// grows with the number of cells it walks.
///
/// Every total is a sum of terms of order one per cell, by three exact
/// identities that all follow from the stationarity condition
/// `omega_c = v k_c - d_c`:
///
/// * The offset residual `sum_c omega_c - v s` is `-sum_c d_c`.
/// * For an expressed cell, `k ln w - k ln(k / s) = k ln(omega / (v k)) =
///   k ln(1 - d / (v k))`; the anchor `sum_c k_c ln(k_c / s)` is the same in
///   every bin.
/// * For an expressed cell, `ln(1 + omega) - ln(v k) =
///   ln(1 + (1 - d) / (v k))`; the anchor `sum_c ln(v k_c)` is known to the
///   host exactly.
///
/// The untransformed sums are of order `v s` and `K ln K`, which for a gene
/// holding a quarter of the reads put `f32` rounding at units in the log
/// likelihood.
///
/// With the table, a gene lists only its nonzero cells. Each also removes its
/// own empty-argument terms, `omega(ln T_c + shift)`, which the table's
/// all-cell sums include; the table is added after the reduction. Those terms
/// depend on the cell only through its total, so the removed share is about the
/// gene's density times the table's, and the difference cancels by at most that
/// factor. Without the table a gene walks its dense row instead.
///
/// ### Params
///
/// * `counts` - Dense counts, gene-major, for a gene without the table.
/// * `list_offsets` - Start of each gene's listed cells, length `n_genes + 1`.
/// * `list_cells` - Cell index of each listed cell.
/// * `list_counts` - Count of each listed cell.
/// * `log_totals` - `ln T_c` per cell.
/// * `n_cells` - Number of cells.
/// * `gene` - This gene.
/// * `use_table` - Whether this gene's empty cells come from the table.
/// * `table` - The table's four tensors and panel count; see [`table_sums`].
/// * `partials` - Shared scratch, [`N_TOTALS`]` * `[`MAX_PLANES_PER_GENE`].
/// * `v` - The variance bin.
/// * `shift` - `ln(v s) - z`.
/// * `tot` - Output, five totals: `sum d`, `S_A = sum omega / (1 + omega)`,
///   `sum d^2`, `sum_{k > 0} k ln(omega / (v k))` and
///   `sum_{k = 0} ln(1 + omega) + sum_{k > 0} ln((1 + omega) / (v k))`.
///   Without the table's share for a table gene.
/// * `tab` - Output, the table's share of each total, in `f32`; zero for a
///   gene without the table. Kept apart so the host can replace it with the
///   same share in `f64`: the solve only needs its sign and slope, but the
///   likelihood takes the off-root term `sum d / v`, which carries `f32`
///   rounding of `G_omega` times `s`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn sweep<F: Float>(
    counts: &Tensor<F>,
    list_offsets: &Tensor<u32>,
    list_cells: &Tensor<u32>,
    list_counts: &Tensor<F>,
    log_totals: &Tensor<F>,
    n_cells: u32,
    gene: u32,
    use_table: bool,
    table_breaks: &Tensor<F>,
    table_anchors: &Tensor<F>,
    table_coeffs: &Tensor<F>,
    table_scalars: &Tensor<F>,
    n_panels: u32,
    partials: &mut SharedMemory<F>,
    v: F,
    shift: F,
    tot: &mut Array<F>,
    tab: &mut Array<F>,
) {
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let mut sum_d = zero;
    let mut curvature = zero;
    let mut sum_sq = zero;
    let mut sum_kw = zero;
    let mut sum_log1p = zero;

    let start = list_offsets[gene as usize];
    let mut n = n_cells;
    if use_table {
        n = list_offsets[(gene + 1u32) as usize] - start;
    }
    let row = gene * n_cells;
    let mut j = UNIT_POS_X;
    while j < n {
        let mut k = counts[(row + j) as usize];
        let mut lt = log_totals[j as usize];
        if use_table {
            k = list_counts[(start + j) as usize];
            lt = log_totals[list_cells[(start + j) as usize] as usize];
        }
        let x = v * k + lt + shift;
        let t = log_omega::<F>(x);
        let w = omega_from_log::<F>(x, t);
        let d = log_fold_change::<F>(k, x, t, w, v, lt, shift);
        sum_d += d;
        curvature += w / (one + w);
        sum_sq += d * d;
        if k > zero {
            let vk = v * k;
            sum_kw += k * ln_near_one::<F>((zero - d) / vk, w / vk);
            sum_log1p += ln_near_one::<F>((one - d) / vk, (one + w) / vk);
        } else {
            sum_log1p += log1p::<F>(w);
        }
        if use_table && k > zero {
            let x0 = lt + shift;
            let w0 = omega_from_log::<F>(x0, log_omega::<F>(x0));
            // An empty cell's `d` is `-omega`, so removing its term adds `omega`.
            sum_d += w0;
            curvature -= w0 / (one + w0);
            sum_sq -= w0 * w0;
            sum_log1p -= log1p::<F>(w0);
        }
        j += CUBE_DIM_X;
    }

    let plane_d = plane_sum(sum_d);
    let plane_curvature = plane_sum(curvature);
    let plane_sq = plane_sum(sum_sq);
    let plane_kw = plane_sum(sum_kw);
    let plane_log1p = plane_sum(sum_log1p);
    // The plane's own id, never `UNIT_POS_X / PLANE_DIM`: nothing obliges a
    // driver to lay subgroups out contiguously.
    if UNIT_POS_PLANE == 0u32 {
        partials[(PLANE_POS * N_TOTALS) as usize] = plane_d;
        partials[(PLANE_POS * N_TOTALS + 1u32) as usize] = plane_curvature;
        partials[(PLANE_POS * N_TOTALS + 2u32) as usize] = plane_sq;
        partials[(PLANE_POS * N_TOTALS + 3u32) as usize] = plane_kw;
        partials[(PLANE_POS * N_TOTALS + 4u32) as usize] = plane_log1p;
    }
    sync_cube();

    let n_planes = CUBE_DIM_X / PLANE_DIM;
    let mut q = 0u32;
    while q < N_TOTALS {
        tot[q as usize] = zero;
        q += 1u32;
    }
    let mut p = 0u32;
    while p < n_planes {
        let mut q = 0u32;
        while q < N_TOTALS {
            tot[q as usize] += partials[(p * N_TOTALS + q) as usize];
            q += 1u32;
        }
        p += 1u32;
    }
    // Nobody may overwrite the partials until every thread has read them.
    sync_cube();

    let mut q = 0u32;
    while q < N_TOTALS {
        tab[q as usize] = zero;
        q += 1u32;
    }
    if use_table {
        table_sums::<F>(
            table_breaks,
            table_anchors,
            table_coeffs,
            table_scalars,
            n_panels,
            shift,
            tab,
        );
    }
}

/// Add the table's all-cell empty-argument sums at `shift` to the totals.
///
/// The device twin of `crate::model::empty_cells::ShiftTable::eval`, in `f32`:
/// the asymptotic form below the table, the panel's Chebyshev series inside it.
/// Above the table the shift is clamped to its top. No root lies there, and the
/// clamped sums keep the residual's sign: the table's top was chosen so that
/// `G_omega` there exceeds every `v s` of the run, and a nonzero cell's own
/// `omega` is at least its empty-argument one, so `sum_c omega_c - v s` stays
/// positive and the bracket still moves the right way.
///
/// ### Params
///
/// * `breaks` - Panel boundaries, length `n_panels + 1`.
/// * `anchors` - `G_f` at each panel's midpoint, `[panel * N_SUMS + f]`.
/// * `coeffs` - Chebyshev coefficients of `ln(G_f / G_f(mid))`,
///   `[(panel * N_SUMS + f) * PANEL_POINTS + k]`.
/// * `scalars` - `ln(sum_c T_c)`, then `ln(sum_c T_c^2)`.
/// * `n_panels` - Number of panels.
/// * `shift` - `ln(v s) - z`.
/// * `tot` - Five totals laid out as [`sweep`]'s; four are added to.
#[cube]
#[allow(clippy::too_many_arguments)]
fn table_sums<F: Float>(
    breaks: &Tensor<F>,
    anchors: &Tensor<F>,
    coeffs: &Tensor<F>,
    scalars: &Tensor<F>,
    n_panels: u32,
    shift: F,
    tot: &mut Array<F>,
) {
    let two = F::new(2.0_f32);
    if shift < breaks[0] {
        let w = F::exp(shift + scalars[0]);
        tot[0] -= w;
        tot[1] += w;
        tot[2] += F::exp(two * shift + scalars[1]);
        tot[4] += w;
    } else {
        let s = F::min(shift, breaks[n_panels as usize]);
        // Largest `lo` with `breaks[lo] <= s`, at most the last panel.
        let mut lo = 0u32;
        let mut hi = n_panels;
        while hi - lo > 1u32 {
            let mid = (lo + hi) / 2u32;
            if breaks[mid as usize] <= s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let a = breaks[lo as usize];
        let b = breaks[(lo + 1u32) as usize];
        let t = (two * s - a - b) / (b - a);

        let mut f = 0u32;
        while f < N_SUMS as u32 {
            let base = (lo * N_SUMS as u32 + f) * PANEL_POINTS as u32;
            // Clenshaw's recurrence, as `crate::utils::chebyshev::clenshaw`.
            let mut b1 = F::new(0.0_f32);
            let mut b2 = F::new(0.0_f32);
            // Counted up from a literal: a counter initialised from a
            // comptime constant is immutable to the cube expansion.
            let mut i = 1u32;
            while i < PANEL_POINTS as u32 {
                let k = PANEL_POINTS as u32 - i;
                let b0 = two * t * b1 - b2 + coeffs[(base + k) as usize];
                b2 = b1;
                b1 = b0;
                i += 1u32;
            }
            let series = t * b1 - b2 + coeffs[base as usize];
            let g = anchors[(lo * N_SUMS as u32 + f) as usize] * F::exp(series);
            if f == 0u32 {
                tot[0] -= g;
            } else if f == 1u32 {
                tot[1] += g;
            } else if f == 2u32 {
                tot[2] += g;
            } else {
                tot[4] += g;
            }
            f += 1u32;
        }
    }
}

/// The half-unit drop of SI eq. 38 at a trial `sigma`, minus the half unit.
///
/// ### Params
///
/// * `sigma` - Trial error bar.
/// * `d` - The cell's log fold change.
/// * `v` - The variance bin.
/// * `s` - `K`, the gene's total count.
/// * `wn` - The cell's normalised weight, `omega / (v s)`.
///
/// ### Returns
///
/// The drop in log posterior minus one half; its root is the error bar.
#[cube]
fn half_unit_drop<F: Float>(sigma: F, d: F, v: F, s: F, wn: F) -> F {
    let two = F::new(2.0_f32);
    sigma * (two * d + sigma) / (two * v) + s * log1p::<F>(wn * expm1::<F>(sigma))
        - F::new(HALF_UNIT_DROP)
}

/// Posterior variance of an empty cell's log fold change, SI eq. 38.
///
/// Mirrors `crate::model::variance::empty_cell_variance`: bracket by doubling
/// from the Gaussian width, then Newton safeguarded by bisection.
///
/// ### Params
///
/// * `d` - The cell's log fold change.
/// * `w` - `omega` for the cell.
/// * `v` - The variance bin.
/// * `s` - `K`, the gene's total count.
///
/// ### Returns
///
/// `sigma^2`.
#[cube]
fn empty_cell_variance<F: Float>(d: F, w: F, v: F, s: F) -> F {
    let one = F::new(1.0_f32);
    let zero = F::new(0.0_f32);
    let wn = w / (v * s);
    let seed = F::sqrt(v / (one + w));

    let mut lo = zero;
    let mut hi = seed;
    let mut brackets = 0u32;
    while half_unit_drop::<F>(hi, d, v, s, wn) < zero {
        lo = hi;
        hi *= F::new(2.0_f32);
        brackets += 1u32;
        // Past the largest finite `f32` the drop has overflowed to infinity and
        // the bisection below still converges.
        if brackets > SIGMA_MAX_BRACKET || hi > F::new(f32::MAX) {
            break;
        }
    }

    let mut sigma = F::min(seed, hi);
    let mut i = 0u32;
    while i < SIGMA_MAX_ITER {
        let f = half_unit_drop::<F>(sigma, d, v, s, wn);
        if F::abs(f) <= F::new(SIGMA_TOL_F32) {
            break;
        }
        if f < zero {
            lo = sigma;
        } else {
            hi = sigma;
        }
        if hi - lo <= F::new(SIGMA_BRACKET_ULPS_F32 * f32::EPSILON) * sigma {
            break;
        }
        let e = F::exp(sigma);
        let denominator = one + wn * (e - one);
        let slope = (d + sigma) / v + s * wn * e / denominator;
        let next = sigma - f / slope;
        // A NaN or an infinity fails both comparisons and bisects.
        if next > lo && next < hi {
            sigma = next;
        } else {
            sigma = (lo + hi) * F::new(0.5_f32);
        }
        i += 1u32;
    }
    sigma * sigma
}

/// Posterior variance of an expressed cell's log fold change, SI eq. 37.
///
/// Same form as `crate::model::variance::gaussian_variance`.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `w` - `omega` for the cell.
/// * `curvature` - `S_A` for the gene at this bin.
///
/// ### Returns
///
/// The Gaussian posterior variance of `d_c`.
#[cube]
fn gaussian_variance<F: Float>(v: F, w: F, curvature: F) -> F {
    let one = F::new(1.0_f32);
    let one_plus = one + w;
    (v / one_plus) * (one + w * w / (one_plus * curvature))
}

/////////////
// Kernels //
/////////////

/// First pass: solve the offset in every bin and reduce the likelihood terms.
///
/// SI eq. 27 per bin, with the same bracket-then-Newton solve as
/// `crate::model::fractions::solve_stationary`, warm started from the previous
/// bin's offset. One workgroup owns one gene and every thread runs the whole
/// bin loop on identical reduced totals, so the threads never disagree about
/// the next step. The plane width is read at run time; the workgroup may hold
/// any whole number of planes up to [`MAX_PLANES_PER_GENE`].
///
/// The log marginal likelihood itself is assembled on the host in `f64`: this
/// kernel returns only its bin-dependent pieces, none of which cancel.
///
/// ### Params
///
/// * `counts` - Dense counts, `[gene * n_cells + c]`, read for genes without
///   the table.
/// * `list_offsets` - Start of each gene's listed cells, length `n_genes + 1`.
/// * `list_cells` - Cell index of each listed cell.
/// * `list_counts` - Count of each listed cell.
/// * `log_totals` - `ln T_c`, length `n_cells`.
/// * `gene_scalars` - `[r * n_genes + gene]`: `K`, then the first bin's guess
///   for `zeta`, then `1` if the gene's empty cells come from the table and
///   only its nonzero cells are listed, `0` if it walks its dense row.
/// * `table_breaks`, `table_anchors`, `table_coeffs`, `table_scalars`,
///   `n_panels` - The empty-cell table; see [`table_sums`]. Placeholders when
///   no gene uses it.
/// * `bins` - `[(r * n_genes + gene) * n_bins + b]`: `v`, then `ln(v s) - a_b`,
///   then the anchor step `a_b - a_{b-1}` (zero at the first bin).
/// * `out` - `[(r * n_genes + gene) * n_bins + b]`: `zeta`, then totals one to
///   four of `sweep`, then total zero, the residual `sum d` the solve
///   stopped at. All at the converged offset. For a table gene, total one
///   (`S_A`, which the second pass reads) includes the table's share and the
///   others do not; the host adds theirs in `f64`.
/// * `status` - Per gene: `0` if every bin converged, else `1 +` the first bin
///   that did not. Bins after it are left unwritten.
/// * `n_genes` - Genes in the launch.
/// * `n_cells` - Cells.
/// * `n_bins` - Bins per gene.
/// * `cold_start` - Non-zero if the first bin's guess is cold.
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> gene
/// * `UNIT_POS_X` -> lane, striding over the cells by `CUBE_DIM_X`
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn sweep_grid_gpu<F: Float + CubeElement>(
    counts: &Tensor<F>,
    list_offsets: &Tensor<u32>,
    list_cells: &Tensor<u32>,
    list_counts: &Tensor<F>,
    log_totals: &Tensor<F>,
    gene_scalars: &Tensor<F>,
    table_breaks: &Tensor<F>,
    table_anchors: &Tensor<F>,
    table_coeffs: &Tensor<F>,
    table_scalars: &Tensor<F>,
    n_panels: u32,
    bins: &Tensor<F>,
    out: &mut Tensor<F>,
    status: &mut Tensor<u32>,
    n_genes: u32,
    n_cells: u32,
    n_bins: u32,
    cold_start: u32,
) {
    let gene = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if gene >= n_genes {
        terminate!();
    }
    let lead = UNIT_POS_X == 0u32;
    let mut partials = SharedMemory::<F>::new((N_TOTALS * MAX_PLANES_PER_GENE) as usize);
    let zero = F::new(0.0_f32);
    let use_table = gene_scalars[(2u32 * n_genes + gene) as usize] > zero;
    let one = F::new(1.0_f32);

    // `z` below is the anchored offset `zeta = z - a_b`; see [`OFFSET_ULPS_F32`].
    let mut z = gene_scalars[(n_genes + gene) as usize];
    let mut tot = Array::<F>::new(N_TOTALS as usize);
    let mut tab = Array::<F>::new(N_TOTALS as usize);
    let mut failed: u32 = 0u32;

    let mut b = 0u32;
    while b < n_bins {
        let v = bins[(gene * n_bins + b) as usize];
        let log_vs = bins[((n_genes + gene) * n_bins + b) as usize];
        // Carry the previous bin's solution into this bin's anchor frame.
        z -= bins[((2u32 * n_genes + gene) * n_bins + b) as usize];
        let resolution = F::new(OFFSET_ULPS_F32 * f32::EPSILON);

        // `F(z) = sum_c omega_c - v s = -sum_c d_c`; see `sweep`.
        sweep::<F>(
            counts,
            list_offsets,
            list_cells,
            list_counts,
            log_totals,
            n_cells,
            gene,
            use_table,
            table_breaks,
            table_anchors,
            table_coeffs,
            table_scalars,
            n_panels,
            &mut partials,
            v,
            log_vs - z,
            &mut tot,
            &mut tab,
        );
        let mut f = zero - (tot[0] + tab[0]);

        // `F` is strictly decreasing in `z`, so a positive residual means `z`
        // is too small. Expand outwards by doubling until the sign flips.
        let mut lo = z;
        let mut hi = z;
        let ascending = f > zero;
        let mut step = F::new(OFFSET_BRACKET_STEP_WARM);
        if cold_start != 0u32 && b == 0u32 {
            step = F::new(OFFSET_BRACKET_STEP_COLD);
        }
        let mut n_bracket = 0u32;
        loop {
            if ascending {
                hi = lo + step;
                z = hi;
            } else {
                lo = hi - step;
                z = lo;
            }
            sweep::<F>(
                counts,
                list_offsets,
                list_cells,
                list_counts,
                log_totals,
                n_cells,
                gene,
                use_table,
                table_breaks,
                table_anchors,
                table_coeffs,
                table_scalars,
                n_panels,
                &mut partials,
                v,
                log_vs - z,
                &mut tot,
                &mut tab,
            );
            f = zero - (tot[0] + tab[0]);
            if (ascending && f <= zero) || (!ascending && f >= zero) {
                break;
            }
            if ascending {
                lo = hi;
            } else {
                hi = lo;
            }
            step *= F::new(2.0_f32);
            n_bracket += 1u32;
            if n_bracket > OFFSET_MAX_BRACKET {
                failed = 1u32;
                break;
            }
        }

        if failed == 0u32 {
            let mut iteration = 0u32;
            loop {
                // The Newton step `f / S_A` is below the offset's resolution.
                if F::abs(f) <= resolution * (one + F::abs(z)) * (tot[1] + tab[1]) {
                    break;
                }
                if f > zero {
                    lo = z;
                } else {
                    hi = z;
                }
                if hi - lo <= resolution * (one + F::abs(z)) {
                    break;
                }
                iteration += 1u32;
                if iteration >= OFFSET_MAX_ITER {
                    failed = 1u32;
                    break;
                }
                // F'(z) = -S_A, which the sweep at `z` already reduced.
                let next = z + f / (tot[1] + tab[1]);
                if next > lo && next < hi {
                    z = next;
                } else {
                    z = (lo + hi) * F::new(0.5_f32);
                }
                sweep::<F>(
                    counts,
                    list_offsets,
                    list_cells,
                    list_counts,
                    log_totals,
                    n_cells,
                    gene,
                    use_table,
                    table_breaks,
                    table_anchors,
                    table_coeffs,
                    table_scalars,
                    n_panels,
                    &mut partials,
                    v,
                    log_vs - z,
                    &mut tot,
                    &mut tab,
                );
                f = zero - (tot[0] + tab[0]);
            }
        }

        if failed != 0u32 {
            break;
        }

        if lead {
            let base = gene * n_bins + b;
            let stride = n_genes * n_bins;
            out[base as usize] = z;
            out[(stride + base) as usize] = tot[1] + tab[1];
            out[(2u32 * stride + base) as usize] = tot[2];
            out[(3u32 * stride + base) as usize] = tot[3];
            out[(4u32 * stride + base) as usize] = tot[4];
            out[(5u32 * stride + base) as usize] = tot[0];
        }
        b += 1u32;
    }

    if lead {
        let mut code = 0u32;
        if failed != 0u32 {
            code = b + 1u32;
        }
        status[gene as usize] = code;
    }
}

/// Second pass: integrate each cell's estimate over the kept bins.
///
/// SI eq. 39 and 42, as in `crate::model::gene::marginalise`, with the spread
/// of `d*_c` across bins accumulated by weighted Welford. Each bin's offset and
/// `S_A` come from [`fn@sweep_grid_gpu`]'s output, still on the device, so no
/// bin needs solving again. Cells are independent here: one thread per cell,
/// no reduction and no barrier. A point-estimate rule is the same kernel over
/// one bin of weight one, where the spread term vanishes.
///
/// ### Params
///
/// * `counts` - Dense counts, `[gene * n_cells + c]`.
/// * `log_totals` - `ln T_c`, length `n_cells`.
/// * `gene_scalars` - As for [`fn@sweep_grid_gpu`].
/// * `bins` - As for [`fn@sweep_grid_gpu`].
/// * `sweep_out` - [`fn@sweep_grid_gpu`]'s output.
/// * `weights` - `W_b`, `[gene * n_bins + b]`; zero for a bin the host dropped.
/// * `out` - `[r * n_genes * n_cells + gene * n_cells + c]`: `d_c`, then `e_c`.
/// * `n_genes` - Genes in the launch.
/// * `n_cells` - Cells.
/// * `n_bins` - Bins per gene.
/// * `blocks_per_gene` - Cubes covering one gene's cells.
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> block; `block / blocks_per_gene`
///   -> gene
/// * `(block % blocks_per_gene) * CUBE_DIM_X + UNIT_POS_X` -> cell
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn marginalise_gpu<F: Float + CubeElement>(
    counts: &Tensor<F>,
    log_totals: &Tensor<F>,
    gene_scalars: &Tensor<F>,
    bins: &Tensor<F>,
    sweep_out: &Tensor<F>,
    weights: &Tensor<F>,
    out: &mut Tensor<F>,
    n_genes: u32,
    n_cells: u32,
    n_bins: u32,
    blocks_per_gene: u32,
) {
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let gene = block / blocks_per_gene;
    let c = (block % blocks_per_gene) * CUBE_DIM_X + UNIT_POS_X;
    if gene >= n_genes || c >= n_cells {
        terminate!();
    }
    let zero = F::new(0.0_f32);

    let k = counts[(gene * n_cells + c) as usize];
    let lt = log_totals[c as usize];
    let s = gene_scalars[gene as usize];
    let stride = n_genes * n_bins;

    let mut running = zero;
    let mut mean_d = zero;
    let mut m2_d = zero;
    let mut mean_var = zero;

    let mut b = 0u32;
    while b < n_bins {
        let base = gene * n_bins + b;
        let weight = weights[base as usize];
        if weight > zero {
            let v = bins[base as usize];
            // Both anchored by the same `a_b`, which cancels in the shift.
            let log_vs = bins[(stride + base) as usize];
            let z = sweep_out[base as usize];
            let curvature = sweep_out[(stride + base) as usize];
            let shift = log_vs - z;

            let x = v * k + lt + shift;
            let t = log_omega::<F>(x);
            let w = omega_from_log::<F>(x, t);
            let d = log_fold_change::<F>(k, x, t, w, v, lt, shift);
            let mut var = gaussian_variance::<F>(v, w, curvature);
            if k <= zero {
                var = empty_cell_variance::<F>(d, w, v, s);
            }

            running += weight;
            let share = weight / running;
            let delta = d - mean_d;
            mean_d += share * delta;
            m2_d += weight * delta * (d - mean_d);
            mean_var += share * (var - mean_var);
        }
        b += 1u32;
    }

    out[(gene * n_cells + c) as usize] = mean_d;
    out[((n_genes + gene) * n_cells + c) as usize] = F::sqrt(mean_var + m2_d / running);
}
