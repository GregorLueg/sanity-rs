//! Sums over cells with no counts, as functions of one scalar.
//!
//! A cell with `k_c = 0` has `x_c = ln T_c + shift`, with
//! `shift = ln(v s) - z`, and stationarity gives it `d_c = -omega(x_c)`. Every
//! reduction the offset solve and the Laplace fit take over such cells is
//! therefore `sum_c f(omega(ln T_c + shift))`, which depends on the gene only
//! through `shift`. Over *all* cells that sum is a function of the totals
//! alone, `G_f(shift)`, and the sum over a gene's empty cells is `G_f` minus the
//! same terms at its nonzero cells.
//!
//! [`ShiftTable`] holds `G_f` for the four `f` needed, as piecewise Chebyshev
//! interpolants built once per run. Each panel stores `G_f` at its midpoint and
//! interpolates `ln(G_f / G_f(mid))`, which is of order one, so the
//! interpolant's absolute error is the relative error in `G_f` and no digits
//! go to representing a large logarithm.
//!
//! ### References
//!
//! Trefethen. *Approximation Theory and Approximation Practice.* SIAM (2013).

use rayon::prelude::*;

use crate::utils::chebyshev::{Chebyshev, clenshaw};
use crate::utils::wright_omega::{log_omega, omega_from_log};

////////////
// Consts //
////////////

/// Chebyshev degree of each panel. Even, so the midpoint is a node.
///
/// Measured 2026-09-30 on an M1 Max, simulated totals, `max_vs = 1.5e7`: 35
/// panels and a 0.03 s build at 4000 cells, 30 and 0.11 s at 20000, 26 and
/// 0.94 s at 200000. Worst relative error against the compensated direct sum
/// `1.5e-14`, `3e-14` for `sum omega^2`. One evaluation is about 100 ns, noise
/// against the `O(nnz)` omega solves of the sweep it replaces the empty cells
/// of, so the degree was not tuned further.
const PANEL_DEGREE: usize = 16;

/// Points per panel.
const PANEL_POINTS: usize = PANEL_DEGREE + 1;

/// Number of interpolated sums.
const N_SUMS: usize = 4;

/// Largest tolerated trailing Chebyshev coefficient of `ln(G_f / G_f(mid))`.
///
/// A panel whose last two coefficients exceed this, for any of the four sums,
/// is split in half.
///
/// Measured 2026-09-30 on 2000 simulated cells: the tail of a converged panel
/// sits at `2e-16` to `5e-15`, the rounding floor of the node values, so
/// `1e-15` could not be met and the split never stopped. This is a decade above
/// that floor.
const PANEL_TOL: f64 = 1e-14;

/// Split depth at which a panel is accepted whatever its tail.
///
/// Guards against rounding noise in the node values keeping the tail above
/// [`PANEL_TOL`], which would otherwise double the panel count at every level.
/// `2^-24` of the domain is far below any width the sums need: 2-wide panels
/// already sit at the noise floor.
const PANEL_MAX_DEPTH: usize = 24;

/// How far below zero the largest `x_c` must sit for the asymptotic form.
///
/// There `omega = e^x (1 - e^x + ...)`, so `G_omega = e^shift sum_c T_c`,
/// `G_{w/(1+w)}` and `G_{ln(1+w)}` equal it and `G_{w^2} = e^{2 shift}
/// sum_c T_c^2`, each to relative `e^-40`, about `4e-18`.
const ASYMPTOTE_DEPTH: f64 = 40.0;

/// Factor by which the table's top end overshoots the largest `v s` of the run.
///
/// An offset solve's bracket can step past its root, so the table covers a
/// little more than the roots need. Beyond it the caller sums directly.
const UPPER_HEADROOM: f64 = 4.0;

///////////////
// EmptySums //
///////////////

/// The four sums over a set of cells at one `shift`, all with `k_c = 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EmptySums {
    /// `sum_c omega_c`.
    pub omega: f64,
    /// `sum_c omega_c / (1 + omega_c)`.
    pub curvature: f64,
    /// `sum_c omega_c^2`, which is `sum_c d_c^2` for empty cells.
    pub omega_sq: f64,
    /// `sum_c ln(1 + omega_c)`.
    pub log_diag: f64,
}

impl EmptySums {
    /// The sums in a fixed order, for the table.
    ///
    /// ### Returns
    ///
    /// `[omega, curvature, omega_sq, log_diag]`.
    fn to_array(self) -> [f64; N_SUMS] {
        [self.omega, self.curvature, self.omega_sq, self.log_diag]
    }
}

/// The four sums over every cell at one `shift`, by compensated summation.
///
/// Neumaier summation, so the node values the table is fitted to carry a few
/// ulp of error whatever the cell count. Naive summation leaves noise growing
/// as `sqrt(C) eps`, which the Chebyshev tail then cannot get under
/// [`PANEL_TOL`]. Build time only, so the extra adds cost nothing that matters.
///
/// ### Params
///
/// * `log_totals` - `ln T_c` of the cells to sum over.
/// * `shift` - `ln(v s) - z`.
///
/// ### Returns
///
/// The sums, `O(n_cells)` Wright omega solves.
pub(crate) fn direct_sums(log_totals: &[f64], shift: f64) -> EmptySums {
    let mut sum = [0.0; N_SUMS];
    let mut comp = [0.0; N_SUMS];
    for &lt in log_totals {
        let x = lt + shift;
        let w = omega_from_log(x, log_omega(x));
        let terms = [w, w / (1.0 + w), w * w, w.ln_1p()];
        for f in 0..N_SUMS {
            let t = sum[f] + terms[f];
            comp[f] += if sum[f].abs() >= terms[f].abs() {
                (sum[f] - t) + terms[f]
            } else {
                (terms[f] - t) + sum[f]
            };
            sum[f] = t;
        }
    }
    EmptySums {
        omega: sum[0] + comp[0],
        curvature: sum[1] + comp[1],
        omega_sq: sum[2] + comp[2],
        log_diag: sum[3] + comp[3],
    }
}

////////////////
// ShiftTable //
////////////////

/// `G_f(shift)` for the four sums, over every cell of the run.
///
/// Built once per run and shared read-only across genes.
#[derive(Clone, Debug)]
pub(crate) struct ShiftTable {
    /// Panel boundaries, ascending, length `n_panels + 1`.
    breaks: Vec<f64>,
    /// `G_f` at each panel's midpoint, `[panel * N_SUMS + f]`.
    anchors: Vec<f64>,
    /// Chebyshev coefficients of `ln(G_f / G_f(mid))`,
    /// `[(panel * N_SUMS + f) * PANEL_POINTS + k]`.
    coeffs: Vec<f64>,
    /// `ln(sum_c T_c)`, for the asymptotic branch.
    log_total_sum: f64,
    /// `ln(sum_c T_c^2)`, for the asymptotic branch.
    log_total_sq_sum: f64,
}

impl ShiftTable {
    /// Build the table.
    ///
    /// The bottom end is where the largest `x_c` sits [`ASYMPTOTE_DEPTH`]
    /// below zero; below it [`Self::eval`] uses the asymptotic form. The top
    /// end is where `G_omega` first exceeds `max_vs` by [`UPPER_HEADROOM`].
    /// Since `omega` increases with `x`, `G_omega(shift) <= v s` at every
    /// root, so no root of the run sits above it.
    ///
    /// Panels are split in half until every sum's Chebyshev tail is below
    /// [`PANEL_TOL`]. Node sums run in parallel.
    ///
    /// ### Params
    ///
    /// * `log_totals` - `ln T_c` for every cell.
    /// * `max_vs` - The largest `v s` any solve of the run can target.
    ///
    /// ### Returns
    ///
    /// The table.
    pub(crate) fn new(log_totals: &[f64], max_vs: f64) -> Self {
        let max_lt = log_totals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_total_sum = log_totals.iter().map(|lt| lt.exp()).sum::<f64>().ln();
        let log_total_sq_sum = log_totals
            .iter()
            .map(|lt| (2.0 * lt).exp())
            .sum::<f64>()
            .ln();

        let lower = -ASYMPTOTE_DEPTH - max_lt;
        let target = UPPER_HEADROOM * max_vs;
        let mut width = 1.0;
        while direct_sums(log_totals, lower + width).omega < target {
            width *= 2.0;
        }
        let upper = lower + width;

        let mut done: Vec<(f64, f64, [f64; N_SUMS], Vec<f64>)> = Vec::new();
        let mut pending = vec![(lower, upper, 0usize)];
        while !pending.is_empty() {
            let fitted: Vec<_> = pending
                .par_iter()
                .map(|&(a, b, depth)| (a, b, depth, fit_panel(log_totals, a, b)))
                .collect();
            pending = Vec::new();
            for (a, b, depth, (anchor, coeffs)) in fitted {
                let converged = coeffs
                    .chunks_exact(PANEL_POINTS)
                    .all(|c| c[PANEL_DEGREE - 1].abs().max(c[PANEL_DEGREE].abs()) <= PANEL_TOL);
                if converged || depth >= PANEL_MAX_DEPTH {
                    done.push((a, b, anchor, coeffs));
                } else {
                    let mid = 0.5 * (a + b);
                    pending.push((a, mid, depth + 1));
                    pending.push((mid, b, depth + 1));
                }
            }
        }
        done.sort_by(|p, q| p.0.total_cmp(&q.0));

        let mut breaks = Vec::with_capacity(done.len() + 1);
        let mut anchors = Vec::with_capacity(done.len() * N_SUMS);
        let mut coeffs = Vec::with_capacity(done.len() * N_SUMS * PANEL_POINTS);
        for (a, _, anchor, c) in &done {
            breaks.push(*a);
            anchors.extend_from_slice(anchor);
            coeffs.extend_from_slice(c);
        }
        breaks.push(upper);

        Self {
            breaks,
            anchors,
            coeffs,
            log_total_sum,
            log_total_sq_sum,
        }
    }

    /// Lowest and highest `shift` the table covers.
    ///
    /// ### Returns
    ///
    /// `(lower, upper)`.
    #[cfg(test)]
    pub(crate) fn domain(&self) -> (f64, f64) {
        (self.breaks[0], self.breaks[self.breaks.len() - 1])
    }

    /// `G_f(shift)` for the four sums over every cell.
    ///
    /// ### Params
    ///
    /// * `shift` - `ln(v s) - z`.
    ///
    /// ### Returns
    ///
    /// The sums, or `None` above the table, where the caller must sum directly.
    #[inline]
    pub(crate) fn eval(&self, shift: f64) -> Option<EmptySums> {
        let lower = self.breaks[0];
        if shift < lower {
            let omega = (shift + self.log_total_sum).exp();
            return Some(EmptySums {
                omega,
                curvature: omega,
                omega_sq: (2.0 * shift + self.log_total_sq_sum).exp(),
                log_diag: omega,
            });
        }
        let n_panels = self.breaks.len() - 1;
        if shift > self.breaks[n_panels] || shift.is_nan() {
            return None;
        }
        let p = (self.breaks.partition_point(|&b| b <= shift) - 1).min(n_panels - 1);
        let (a, b) = (self.breaks[p], self.breaks[p + 1]);
        let t = (2.0 * shift - a - b) / (b - a);

        let base = p * N_SUMS * PANEL_POINTS;
        let mut g = [0.0; N_SUMS];
        for (f, out) in g.iter_mut().enumerate() {
            let c = &self.coeffs[base + f * PANEL_POINTS..base + (f + 1) * PANEL_POINTS];
            *out = self.anchors[p * N_SUMS + f] * clenshaw(c, t).exp();
        }
        Some(EmptySums {
            omega: g[0],
            curvature: g[1],
            omega_sq: g[2],
            log_diag: g[3],
        })
    }
}

/////////////
// Helpers //
/////////////

/// Fit one panel.
///
/// Samples the four sums at the Chebyshev extrema of `[a, b]` and takes the
/// discrete cosine transform of `ln(G_f / G_f(mid))`. The ratio is formed
/// before the logarithm, so the node values carry an absolute error of a few
/// ulp whatever the size of `G_f`.
///
/// ### Params
///
/// * `log_totals` - `ln T_c` for every cell.
/// * `a` - Panel start.
/// * `b` - Panel end.
///
/// ### Returns
///
/// The four midpoint anchors and the coefficients, `[f * PANEL_POINTS + k]`.
fn fit_panel(log_totals: &[f64], a: f64, b: f64) -> ([f64; N_SUMS], Vec<f64>) {
    let cheb = Chebyshev::new(a, b, PANEL_DEGREE);
    let nodes: Vec<[f64; N_SUMS]> = (0..PANEL_POINTS)
        .into_par_iter()
        .map(|j| direct_sums(log_totals, cheb.node(j)).to_array())
        .collect();
    let anchor = nodes[PANEL_DEGREE / 2];

    let mut coeffs = vec![0.0; N_SUMS * PANEL_POINTS];
    let mut y = [0.0; PANEL_POINTS];
    for (f, c) in coeffs.chunks_exact_mut(PANEL_POINTS).enumerate() {
        for (yj, g) in y.iter_mut().zip(&nodes) {
            *yj = (g[f] / anchor[f]).ln();
        }
        cheb.coefficients(&y, c);
    }
    (anchor, coeffs)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::{SimulationParams, simulate};

    /// Log totals of a simulated run.
    fn log_totals(n_cells: usize, library_size: f64) -> Vec<f64> {
        let sim = simulate(Some(SimulationParams {
            n_genes: 200,
            n_cells,
            library_size,
            ..SimulationParams::default()
        }))
        .expect("valid simulation");
        sim.cell_totals.iter().map(|t| t.ln()).collect()
    }

    /// Worst relative error of each sum over `n` shifts spread across `[lo, hi]`.
    fn worst_error(table: &ShiftTable, lt: &[f64], lo: f64, hi: f64, n: usize) -> [f64; N_SUMS] {
        let mut worst = [0.0f64; N_SUMS];
        for i in 0..n {
            // Irrational stride so no sample lands on a node.
            let u = (i as f64 * 0.618_033_988_749_895).fract();
            let shift = lo + (hi - lo) * u;
            let got = table.eval(shift).expect("inside the table").to_array();
            let want = direct_sums(lt, shift).to_array();
            for f in 0..N_SUMS {
                worst[f] = worst[f].max(((got[f] - want[f]) / want[f]).abs());
            }
        }
        worst
    }

    #[test]
    fn test_shift_table_matches_direct_sum() {
        let lt = log_totals(2000, 500.0);
        let table = ShiftTable::new(&lt, 50.0 * 2e4);
        let (lo, hi) = table.domain();
        let worst = worst_error(&table, &lt, lo - 5.0, hi, 2000);
        for (f, e) in worst.iter().enumerate() {
            assert!(*e < 1e-13, "sum {f}: relative error {e:e}");
        }
    }

    #[test]
    fn test_shift_table_is_none_above_its_domain() {
        let lt = log_totals(500, 500.0);
        let table = ShiftTable::new(&lt, 1e3);
        let (_, hi) = table.domain();
        assert!(table.eval(hi + 1.0).is_none());
        assert!(table.eval(hi - 1e-9).is_some());
    }
}
