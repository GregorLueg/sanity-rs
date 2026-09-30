//! Chebyshev interpolation on the extrema grid.
//!
//! Samples at `cos(pi j / n)`, coefficients by the type I discrete cosine
//! transform, evaluation by Clenshaw's recurrence.
//!
//! ### References
//!
//! Trefethen. *Approximation Theory and Approximation Practice.* SIAM (2013).

use std::f64::consts::PI;

/// Interpolation on `[a, b]` at a fixed degree, with the transform tabulated.
#[derive(Clone, Debug)]
pub(crate) struct Chebyshev {
    /// Interval start.
    a: f64,
    /// Interval end.
    b: f64,
    /// Degree `n`; there are `n + 1` nodes.
    degree: usize,
    /// `cos(pi j k / n)` with the transform's end weights folded in,
    /// `[k * (n + 1) + j]`.
    transform: Vec<f64>,
}

impl Chebyshev {
    /// Set up interpolation of degree `degree` on `[a, b]`.
    ///
    /// ### Params
    ///
    /// * `a` - Interval start.
    /// * `b` - Interval end, above `a`.
    /// * `degree` - Polynomial degree, at least one.
    ///
    /// ### Returns
    ///
    /// The interpolator.
    pub(crate) fn new(a: f64, b: f64, degree: usize) -> Self {
        let n = degree as f64;
        let points = degree + 1;
        let mut transform = vec![0.0; points * points];
        for k in 0..points {
            let end_k = if k == 0 || k == degree { 0.5 } else { 1.0 };
            for j in 0..points {
                let end_j = if j == 0 || j == degree { 0.5 } else { 1.0 };
                transform[k * points + j] =
                    end_k * end_j * (2.0 / n) * (PI * (j * k) as f64 / n).cos();
            }
        }
        Self {
            a,
            b,
            degree,
            transform,
        }
    }

    /// Number of nodes, `degree + 1`.
    ///
    /// ### Returns
    ///
    /// The node count.
    pub(crate) fn n_points(&self) -> usize {
        self.degree + 1
    }

    /// Node `j`, mapped onto `[a, b]`.
    ///
    /// ### Params
    ///
    /// * `j` - Node index, below [`Self::n_points`]. Node `degree / 2` is the
    ///   midpoint when the degree is even.
    ///
    /// ### Returns
    ///
    /// The node.
    pub(crate) fn node(&self, j: usize) -> f64 {
        let t = (PI * j as f64 / self.degree as f64).cos();
        0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * t
    }

    /// Chebyshev coefficients of the values at the nodes.
    ///
    /// ### Params
    ///
    /// * `values` - Function values at the nodes, in node order.
    /// * `out` - Output, the coefficients, same length.
    pub(crate) fn coefficients(&self, values: &[f64], out: &mut [f64]) {
        let points = self.n_points();
        for (k, c) in out.iter_mut().enumerate().take(points) {
            let row = &self.transform[k * points..(k + 1) * points];
            *c = row.iter().zip(values).map(|(w, y)| w * y).sum();
        }
    }

    /// Evaluate a series from [`Self::coefficients`] at a point of `[a, b]`.
    ///
    /// ### Params
    ///
    /// * `c` - The coefficients.
    /// * `x` - The point.
    ///
    /// ### Returns
    ///
    /// The interpolant at `x`.
    #[inline]
    pub(crate) fn eval(&self, c: &[f64], x: f64) -> f64 {
        clenshaw(c, (2.0 * x - self.a - self.b) / (self.b - self.a))
    }
}

/// Evaluate a Chebyshev series by Clenshaw's recurrence.
///
/// ### Params
///
/// * `c` - Coefficients, `c[k]` multiplying `T_k`.
/// * `t` - Point in `[-1, 1]`.
///
/// ### Returns
///
/// `sum_k c[k] T_k(t)`.
#[inline(always)]
pub(crate) fn clenshaw(c: &[f64], t: f64) -> f64 {
    let (mut b1, mut b2) = (0.0, 0.0);
    for &ck in c[1..].iter().rev() {
        let b0 = 2.0 * t * b1 - b2 + ck;
        b2 = b1;
        b1 = b0;
    }
    t * b1 - b2 + c[0]
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_reproduces_a_smooth_function() {
        let cheb = Chebyshev::new(2.0, 9.0, 24);
        let values: Vec<f64> = (0..cheb.n_points()).map(|j| cheb.node(j).exp()).collect();
        let mut c = vec![0.0; cheb.n_points()];
        cheb.coefficients(&values, &mut c);
        for i in 0..100 {
            let x = 2.0 + 7.0 * (i as f64 * 0.618_033_988_749_895).fract();
            // The error is uniform in absolute terms, so scale by the maximum.
            let err = (cheb.eval(&c, x) - x.exp()).abs() / 9.0f64.exp();
            assert!(err < 1e-14, "x = {x}, scaled error {err:e}");
        }
    }
}
