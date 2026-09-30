//! Per-cell posterior variance of the log fold change at a fixed gene variance.
//!
//! Two rules. For cells with counts, the diagonal of `M^{-1}` from the minor
//! over the determinant, which reduces to `omega_c` and the rank-one sum that
//! [`super::likelihood::laplace`] has already accumulated. For cells with no
//! counts the log posterior is asymmetric about its maximum -- zero counts
//! bound the log fold change from above but are consistent with arbitrarily low
//! values -- so the Gaussian width is replaced by the half-unit drop of SI
//! eq. 38.

use super::fractions::Stationary;

////////////
// Consts //
////////////

/// Target drop in log posterior that defines the error bar for empty cells.
///
/// For a Gaussian, `L(mu) - L(mu + sigma) = 1/2`. SI eq. 38 keeps that
/// definition and applies it to the true asymmetric posterior, taking the upper
/// side because that is the side the data constrains.
const HALF_UNIT_DROP: f64 = 0.5;

/// Convergence tolerance on the half-unit-drop residual.
const SIGMA_TOL: f64 = 1e-12;

/// Iteration cap for the half-unit-drop solve.
const SIGMA_MAX_ITER: usize = 60;

/// Cap on the bracket expansion for the half-unit-drop solve.
const SIGMA_MAX_BRACKET: usize = 40;

/// Posterior variance of the log fold change in a cell with at least one count.
///
/// SI eq. 37. With `B_c = s w_c + 1/v = (1 + omega_c) / v` and
/// `R_c / A = omega_c^2 / ((1 + omega_c) S_A)`, the ratio of minor to
/// determinant collapses to a two-operation expression.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `omega` - `omega(x_c)` for this cell.
/// * `curvature_sum` - `S_A = sum_c omega_c / (1 + omega_c)` over all cells.
///
/// ### Returns
///
/// `var(d_c)` at this variance.
#[inline(always)]
pub(crate) fn gaussian_variance(v: f64, omega: f64, curvature_sum: f64) -> f64 {
    let one_plus = 1.0 + omega;
    (v / one_plus) * (1.0 + omega * omega / (one_plus * curvature_sum))
}

/// Posterior variance of the log fold change in a cell with no counts.
///
/// SI eq. 38. Solves
/// `sigma (2 d + sigma) / (2 v) + s ln(1 + w (e^sigma - 1)) = 1/2`
/// for `sigma > 0` and returns `sigma^2`.
///
/// The left-hand side vanishes at `sigma = 0` with zero derivative -- that is
/// the stationarity condition for an empty cell -- and is strictly increasing
/// thereafter, so the root is unique. The second derivative at the origin is
/// `1/v + s w`, so the Gaussian width `sqrt(v / (1 + omega))` is the correct
/// leading-order guess and is used to seed the bracket.
///
/// Seeding from the previous bin's `sigma` instead, since the second pass walks
/// a cell through the grid in order, was measured slower over the
/// whole rule. The grid ascends, so the previous root always
/// undershoots and the bracket loop must double at least once, overshooting to
/// twice it; the Gaussian width is already correct to leading order at the
/// current `v` and needs no expansion at all.
///
/// ### Params
///
/// * `point` - The stationary point for this gene at this variance.
/// * `d` - `d*_c`, the log fold change in this cell.
/// * `omega` - `omega(x_c)` for this cell.
///
/// ### Returns
///
/// `sigma^2`, the symmetric error bar squared.
pub(crate) fn empty_cell_variance(point: &Stationary, d: f64, omega: f64) -> f64 {
    let v = point.v;
    let s = point.s;
    let w = point.weight(omega);

    // `expm1` and `ln_1p` keep the small-sigma regime, where the whole
    // expression is a difference of quantities near one, accurate.
    let drop = |sigma: f64| -> f64 {
        sigma * (2.0 * d + sigma) / (2.0 * v) + s * (w * sigma.exp_m1()).ln_1p() - HALF_UNIT_DROP
    };

    let seed = (v / (1.0 + omega)).sqrt();
    let mut lo = 0.0;
    let mut hi = seed;
    let mut brackets = 0;
    while drop(hi) < 0.0 {
        lo = hi;
        hi *= 2.0;
        brackets += 1;
        if brackets > SIGMA_MAX_BRACKET || !hi.is_finite() {
            // `w (e^sigma - 1)` has overflowed to infinity, so the drop is
            // unbounded above and the bisection below still converges.
            break;
        }
    }

    // No divergence error, unlike the offset solve: `sigma` is bounded by
    // roughly `sqrt(v)`, so the bracket spans at most eight units and bisection
    // alone resolves it to machine precision well inside the iteration cap.
    let mut sigma = seed.min(hi);
    for _ in 0..SIGMA_MAX_ITER {
        let f = drop(sigma);
        if f.abs() <= SIGMA_TOL {
            break;
        }
        if f < 0.0 {
            lo = sigma
        } else {
            hi = sigma
        }

        let e = sigma.exp();
        let denominator = 1.0 + w * (e - 1.0);
        let slope = (d + sigma) / v + s * w * e / denominator;
        let next = sigma - f / slope;

        sigma = if next.is_finite() && next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }

    sigma * sigma
}

/// The posterior variance for one cell, picking the rule from its count.
///
/// ### Params
///
/// * `point` - The stationary point for this gene at this variance.
/// * `count` - The UMI count in this cell.
/// * `d` - `d*_c`, the log fold change in this cell.
/// * `omega` - `omega(x_c)` for this cell.
///
/// ### Returns
///
/// `var(d_c)` at this variance.
#[inline]
pub(crate) fn cell_variance(point: &Stationary, count: f64, d: f64, omega: f64) -> f64 {
    if count > 0.0 {
        gaussian_variance(point.v, omega, point.curvature_sum)
    } else {
        empty_cell_variance(point, d, omega)
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

    /// Diagonal of the inverse curvature matrix, built and inverted densely.
    fn brute_force_diagonal(point: &Stationary, omega: &[f64]) -> Vec<f64> {
        let n = omega.len();
        let mut m = vec![0.0; n * n];
        let mut inv = vec![0.0; n * n];
        for i in 0..n {
            inv[i * n + i] = 1.0;
            for j in 0..n {
                let wi = point.weight(omega[i]);
                let wj = point.weight(omega[j]);
                m[i * n + j] = -point.s * wi * wj;
                if i == j {
                    m[i * n + j] += point.s * wi + 1.0 / point.v;
                }
            }
        }
        for k in 0..n {
            let pivot = m[k * n + k];
            for j in 0..n {
                m[k * n + j] /= pivot;
                inv[k * n + j] /= pivot;
            }
            for i in 0..n {
                if i == k {
                    continue;
                }
                let factor = m[i * n + k];
                for j in 0..n {
                    m[i * n + j] -= factor * m[k * n + j];
                    inv[i * n + j] -= factor * inv[k * n + j];
                }
            }
        }
        (0..n).map(|i| inv[i * n + i]).collect()
    }

    #[test]
    fn test_gaussian_variance_matches_dense_inverse() {
        let counts = vec![1.0, 3.0, 1.0, 5.0, 7.0, 2.0, 2.0, 4.0];
        let totals: Vec<f64> = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0, 450.0, 1330.0, 1010.0];
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let s = counts.iter().sum::<f64>() + 1.0;

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

            let expected = brute_force_diagonal(&point, &omega);
            for c in 0..counts.len() {
                assert_relative_eq!(
                    gaussian_variance(v, omega[c], point.curvature_sum),
                    expected[c],
                    max_relative = 1e-9
                );
            }
        }
    }

    #[test]
    fn test_empty_cell_variance_hits_the_half_unit_drop() {
        let counts = vec![0.0, 3.0, 1.0, 0.0, 7.0, 0.0, 2.0, 0.0];
        let totals: Vec<f64> = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0, 450.0, 1330.0, 1010.0];
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let s = counts.iter().sum::<f64>() + 1.0;

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

            for c in 0..counts.len() {
                if counts[c] > 0.0 {
                    continue;
                }
                let d = point.log_fold_change(log_omega[c], log_totals[c]);
                let sigma = empty_cell_variance(&point, d, omega[c]).sqrt();
                let w = point.weight(omega[c]);
                let drop = sigma * (2.0 * d + sigma) / (2.0 * v) + s * (w * sigma.exp_m1()).ln_1p();
                assert_relative_eq!(drop, 0.5, max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn test_empty_cell_variance_is_bounded_by_the_prior() {
        // The posterior cannot be wider than the prior it came from.
        let counts = vec![0.0, 0.0, 0.0, 0.0, 1.0];
        let totals: Vec<f64> = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0];
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let s = counts.iter().sum::<f64>() + 1.0;

        for &v in &[1e-3, 1.0, 50.0] {
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

            for c in 0..counts.len() {
                if counts[c] > 0.0 {
                    continue;
                }
                let d = point.log_fold_change(log_omega[c], log_totals[c]);
                let var = empty_cell_variance(&point, d, omega[c]);
                assert!(var > 0.0 && var <= v * (1.0 + 1e-9), "v = {v}, var = {var}");
            }
        }
    }
}
