//! The Laplace-approximated marginal likelihood of one gene at one variance.
//!
//! The curvature matrix is diagonal plus rank one, so its determinant follows
//! from the matrix determinant lemma. SI eq. 33 writes the rank-one correction
//! as `1 - sum_c s w_c^2 / (s w_c + 1/v)`, which tends to a difference of
//! nearly equal numbers as `v` grows. The identity used here,
//! `1 - S = sum_c w_c / (1 + v s w_c)`, is exact, manifestly positive and free
//! of that cancellation.
//!
//! Both `w_c` and the diagonal reduce to `omega_c` alone:
//! `v s w_c = omega_c`, so `s w_c + 1/v = (1 + omega_c) / v`. Everything below
//! is therefore a single fused pass over `omega`.

use super::empty_cells::{ShiftTable, direct_sums};
use super::fractions::{NonzeroCells, Stationary};

/////////////
// Laplace //
/////////////

/// Per-gene, per-bin reductions of the Laplace approximation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Laplace {
    /// `ln P(k | v)`, up to an additive constant that is the same in every bin.
    pub log_marginal: f64,
}

/// Evaluate the marginal likelihood at a stationary point.
///
/// One pass over the cells accumulating three reductions at once: the quadratic
/// prior term, the data term and the log determinant's diagonal. The rank-one
/// correction is not among them; the offset solve's Newton derivative is the
/// same sum, so it arrives on the point as `Stationary::curvature_sum` rather
/// than being accumulated again here. Accumulation is in `f64` regardless of
/// storage precision, because the bin-to-bin differences that select a variance
/// are small against the sum over cells.
///
/// ### Params
///
/// * `point` - The stationary point for this gene at this variance.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `omega` - `omega(x_c)` at the stationary point.
/// * `log_omega` - `ln omega(x_c)` at the stationary point.
///
/// ### Returns
///
/// The log marginal likelihood.
pub(crate) fn laplace(
    point: &Stationary,
    counts: &[f64],
    log_totals: &[f64],
    omega: &[f64],
    log_omega: &[f64],
) -> Laplace {
    let mut sum_sq = 0.0;
    let mut sum_data = 0.0;
    let mut sum_log_diag = 0.0;

    for (((&k, &lt), &w), &t) in counts.iter().zip(log_totals).zip(omega).zip(log_omega) {
        let d = point.log_fold_change(t, lt);
        sum_sq += d * d;
        sum_data += k * d;
        sum_log_diag += w.ln_1p();
    }

    combine(point, counts.len(), sum_sq, sum_data, sum_log_diag)
}

/// Evaluate the marginal likelihood from the nonzero cells and the table.
///
/// Same reductions as [`laplace`], with the empty cells' share of each read
/// from the table at the point's `shift`.
///
/// ### Params
///
/// * `point` - The stationary point, as returned by
///   [`super::fractions::solve_stationary_sparse`].
/// * `n_cells` - Number of cells.
/// * `table` - The run's table of empty-cell sums.
/// * `log_totals` - `ln T_c` for every cell, for the direct sum above the table.
/// * `cells` - The gene's nonzero cells, last swept at `point`.
///
/// ### Returns
///
/// The log marginal likelihood.
pub(crate) fn laplace_sparse(
    point: &Stationary,
    n_cells: usize,
    table: &ShiftTable,
    log_totals: &[f64],
    cells: &NonzeroCells,
) -> Laplace {
    let shift = point.log_vs - point.z;
    let all = table
        .eval(shift)
        .unwrap_or_else(|| direct_sums(log_totals, shift));
    let (sum_sq, sum_data, sum_log_diag) = cells.laplace_terms(point);
    combine(
        point,
        n_cells,
        all.omega_sq + sum_sq,
        sum_data,
        all.log_diag + sum_log_diag,
    )
}

/// Assemble the log marginal likelihood from its three cell reductions.
///
/// ### Params
///
/// * `point` - The stationary point.
/// * `n_cells` - Number of cells.
/// * `sum_sq` - `sum_c d_c^2`.
/// * `sum_data` - `sum_c k_c d_c`.
/// * `sum_log_diag` - `sum_c ln(1 + omega_c)`.
///
/// ### Returns
///
/// The log marginal likelihood.
fn combine(
    point: &Stationary,
    n_cells: usize,
    sum_sq: f64,
    sum_data: f64,
    sum_log_diag: f64,
) -> Laplace {
    let n_cells = n_cells as f64;
    let v = point.v;

    // SI eq. 20 at the optimum, using sum_c T_c exp(d*_c) = exp(z).
    let log_star = -0.5 * n_cells * v.ln() - 0.5 * sum_sq / v + sum_data - point.s * point.z;

    // SI eq. 33, in the cancellation-free form. The diagonal contributes
    // sum_c ln((1 + omega_c) / v); the rank-one term contributes ln(S_A / (v s)).
    let log_det = point.curvature_sum.ln() - point.log_vs + sum_log_diag - n_cells * v.ln();

    Laplace {
        log_marginal: log_star - 0.5 * log_det,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::super::fractions::solve_stationary;
    use super::*;
    use approx::assert_relative_eq;

    /// Brute-force `ln det M` from the explicit matrix of SI eq. 31, for a
    /// problem small enough to build it.
    fn brute_force_log_det(point: &Stationary, omega: &[f64]) -> f64 {
        let n = omega.len();
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let wi = point.weight(omega[i]);
                let wj = point.weight(omega[j]);
                m[i * n + j] = -point.s * wi * wj;
                if i == j {
                    m[i * n + j] += point.s * wi + 1.0 / point.v;
                }
            }
        }
        // Gaussian elimination without pivoting; the matrix is symmetric
        // positive definite so no pivoting is needed.
        let mut log_det = 0.0;
        for k in 0..n {
            let pivot = m[k * n + k];
            log_det += pivot.ln();
            for i in (k + 1)..n {
                let factor = m[i * n + k] / pivot;
                for j in k..n {
                    m[i * n + j] -= factor * m[k * n + j];
                }
            }
        }
        log_det
    }

    #[test]
    fn test_log_determinant_matches_dense_matrix() {
        let counts = vec![0.0, 3.0, 1.0, 0.0, 7.0, 0.0, 2.0, 0.0];
        let totals: Vec<f64> = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0, 450.0, 1330.0, 1010.0];
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let s = counts.iter().sum::<f64>() + 1.0;
        let n = counts.len() as f64;

        for &v in &[1e-3, 0.05, 1.0, 12.0, 50.0] {
            let mut omega = vec![0.0; counts.len()];
            let mut log_omega = vec![0.0; counts.len()];
            let point = solve_stationary(
                v,
                s,
                &counts,
                &log_totals,
                totals.iter().sum::<f64>().ln(),
                &mut None,
                &mut omega,
                &mut log_omega,
            )
            .expect("converges");

            let _ = laplace(&point, &counts, &log_totals, &omega, &log_omega);
            let mut sum_log_diag = 0.0;
            for &w in &omega {
                sum_log_diag += w.ln_1p();
            }
            let log_det = point.curvature_sum.ln() - point.log_vs + sum_log_diag - n * v.ln();

            assert_relative_eq!(
                log_det,
                brute_force_log_det(&point, &omega),
                max_relative = 1e-10
            );
        }
    }

    #[test]
    fn test_optimum_beats_its_neighbourhood() {
        // L*(v) must be the maximum of SI eq. 20 over the log fold changes.
        let counts = vec![0.0, 3.0, 1.0, 0.0, 7.0];
        let totals: Vec<f64> = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0];
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let s = counts.iter().sum::<f64>() + 1.0;
        let v = 0.9;

        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let point = solve_stationary(
            v,
            s,
            &counts,
            &log_totals,
            totals.iter().sum::<f64>().ln(),
            &mut None,
            &mut omega,
            &mut log_omega,
        )
        .expect("converges");

        let d: Vec<f64> = (0..counts.len())
            .map(|c| point.log_fold_change(log_omega[c], log_totals[c]))
            .collect();

        let objective = |d: &[f64]| -> f64 {
            let quad: f64 = d.iter().map(|x| x * x).sum();
            let data: f64 = counts.iter().zip(d).map(|(k, x)| k * x).sum();
            let norm: f64 = totals.iter().zip(d).map(|(t, x)| t * x.exp()).sum();
            -0.5 * (counts.len() as f64) * v.ln() - 0.5 * quad / v + data - s * norm.ln()
        };

        let best = objective(&d);
        for c in 0..d.len() {
            for step in [-1e-3, 1e-3] {
                let mut perturbed = d.clone();
                perturbed[c] += step;
                assert!(objective(&perturbed) < best);
            }
        }
    }

    #[test]
    fn test_sparse_sweep_matches_dense() {
        use super::super::fractions::solve_stationary_sparse;
        use crate::simulate::{SimulationParams, simulate};

        let sim = simulate(Some(SimulationParams {
            n_genes: 20,
            n_cells: 500,
            library_size: 200.0,
            ..SimulationParams::default()
        }))
        .expect("valid simulation");
        let n_cells = sim.cell_totals.len();
        let log_totals: Vec<f64> = sim.cell_totals.iter().map(|t| t.ln()).collect();
        let log_total_sum = sim.cell_totals.iter().sum::<f64>().ln();
        let table = ShiftTable::new(&log_totals, 50.0 * 1e5);

        let mut cells = NonzeroCells::new(n_cells);
        let mut counts = vec![0.0; n_cells];
        let mut omega = vec![0.0; n_cells];
        let mut log_omega = vec![0.0; n_cells];
        for g in 0..sim.counts.n_genes() {
            let (indices, values) = sim.counts.gene(g);
            if values.is_empty() {
                continue;
            }
            counts.fill(0.0);
            for (&i, &k) in indices.iter().zip(values) {
                counts[i as usize] = k as f64;
            }
            let s: f64 = counts.iter().sum();
            cells.load(indices, values, &log_totals);

            for &v in &[1e-3, 0.1, 1.0, 10.0, 50.0] {
                let guess = log_total_sum + 0.5 * v;
                let dense = solve_stationary(
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
                // Warm across jumps far larger than a grid step, which is
                // what `log_omega_near`'s cold restart is for.
                let sparse =
                    solve_stationary_sparse(v, s, n_cells, guess, &table, &log_totals, &mut cells)
                        .expect("converges");

                // Both solves stop anywhere inside the residual tolerance, a
                // band `tol / S_A` wide in `z`; this mirrors `solve_offset`'s
                // `tol` with a tenfold margin.
                let tol = v * s * 1e-13_f64.max(4.0 * n_cells as f64 * f64::EPSILON);
                let band = 10.0 * tol / dense.curvature_sum.min(sparse.curvature_sum);
                assert!(
                    (sparse.z - dense.z).abs() <= band,
                    "gene {g}, v {v}: z {} vs {}, band {band:e}",
                    sparse.z,
                    dense.z
                );
                assert_relative_eq!(
                    sparse.curvature_sum,
                    dense.curvature_sum,
                    max_relative = 1e-10
                );
                let want = laplace(&dense, &counts, &log_totals, &omega, &log_omega);
                let got = laplace_sparse(&sparse, n_cells, &table, &log_totals, &cells);
                assert_relative_eq!(
                    got.log_marginal,
                    want.log_marginal,
                    epsilon = 1e-8,
                    max_relative = 1e-12
                );
            }
        }
    }
}
