//! Large cell counts under uniform library sizes.
//!
//! The offset residual is a naive sum over cells, so its rounding noise grows
//! with `C`. Equal `T_c` is the worst case for it, and is what a downsampled or
//! synthetic matrix looks like.

use sanity_sc_rs::config::{SanityParams, VarianceRule, Verbosity};
use sanity_sc_rs::input::CountMatrix;
use sanity_sc_rs::sanity;

/// One gene, counts 1-7 in every third cell, uniform totals.
fn uniform_gene_variance(n_cells: usize) -> f64 {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for c in (0..n_cells).step_by(3) {
        indices.push(c as u32);
        values.push((c % 7 + 1) as u32);
    }
    let indptr = vec![0, indices.len()];
    let counts = CountMatrix::new(indices, values, indptr, n_cells).expect("well formed");
    let totals = vec![5000.0; n_cells];
    let params = SanityParams::new(VarianceRule::Marginalise, 1e-3, 50.0, 21, Verbosity::Quiet);
    let out = sanity::<f64>(&counts, &totals, Some(params)).expect("runs");
    out.variance[0]
}

#[test]
fn test_uniform_totals_scale_invariant() {
    let small = uniform_gene_variance(5_000);
    let large = uniform_gene_variance(20_000);
    assert!((small - 3.3437).abs() < 1e-3, "variance {small}");
    assert!((large - small).abs() < 1e-6, "{small} against {large}");
}

#[test]
#[ignore = "200k cells, about a second in release"]
fn test_uniform_totals_at_two_hundred_thousand_cells() {
    let v = uniform_gene_variance(200_000);
    assert!((v - 3.3437).abs() < 1e-3, "variance {v}");
}
