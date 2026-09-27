//! End-to-end recovery against the simulator's ground truth.
//!
//! The simulator draws from the model's own generative process (SPEC section
//! 10), so these are the conditions under which the method should be at its
//! best. A failure here is a broken implementation, not a modelling limitation.

use sanity_sc_rs::config::{SanityParams, VarianceRule};
use sanity_sc_rs::sanity;
use sanity_sc_rs::simulate::{SimulationParams, simulate};

/// Pearson correlation of two slices.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let mut sa = 0.0;
    let mut sb = 0.0;
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        sa += dx * dx;
        sb += dy * dy;
    }
    num / (sa * sb).sqrt()
}

/// Simulate and run, returning the run and the truth it was drawn from.
fn fixture(
    seed: u64,
    rule: VarianceRule,
) -> (
    sanity_sc_rs::SanityOutput<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let sim = simulate(Some(SimulationParams {
        n_genes: 300,
        n_cells: 400,
        library_size: 10000.0,
        seed,
        ..Default::default()
    }))
    .expect("simulates");

    let params = SanityParams {
        variance_rule: rule,
        ..Default::default()
    };
    let out = sanity::<f64>(&sim.counts, &sim.cell_totals, Some(params)).expect("runs");
    (
        out,
        sim.log_fold_changes,
        sim.variance,
        sim.mean_log_quotient,
    )
}

#[test]
fn test_recovers_log_fold_changes_of_expressed_genes() {
    let (out, truth, _, _) = fixture(1, VarianceRule::Marginalise);
    let sim = simulate(Some(SimulationParams {
        n_genes: 300,
        n_cells: 400,
        library_size: 10000.0,
        seed: 1,
        ..Default::default()
    }))
    .expect("simulates");

    // Correlation with the truth rises with absolute expression, because the
    // information simply is not in the counts for a gene seen a handful of
    // times. Bucket by total UMIs and check the trend as well as the level.
    let mut buckets: Vec<(f64, Vec<f64>)> = vec![
        (100.0, Vec::new()),
        (1000.0, Vec::new()),
        (10000.0, Vec::new()),
    ];
    for g in 0..out.n_genes {
        let row = g * out.n_cells..(g + 1) * out.n_cells;
        let k: f64 = sim.counts.gene(g).1.iter().map(|&x| x as f64).sum();
        let r = correlation(&out.log_fold_changes[row.clone()], &truth[row]);
        for (floor, values) in buckets.iter_mut() {
            if k >= *floor {
                values.push(r);
            }
        }
    }

    let medians: Vec<f64> = buckets
        .iter()
        .map(|(floor, values)| {
            assert!(values.len() > 20, "too few genes above {floor} UMIs");
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("correlations are finite"));
            sorted[sorted.len() / 2]
        })
        .collect();

    assert!(
        medians[0] > 0.6,
        "median correlation above 100 UMIs was {}",
        medians[0]
    );
    assert!(
        medians[1] > 0.75,
        "median correlation above 1000 UMIs was {}",
        medians[1]
    );
    assert!(
        medians[2] > 0.85,
        "median correlation above 10000 UMIs was {}",
        medians[2]
    );
    assert!(
        medians[0] < medians[1] && medians[1] < medians[2],
        "correlation should rise with expression, got {medians:?}"
    );
}

#[test]
fn test_recovers_gene_variances() {
    let (out, _, truth, _) = fixture(2, VarianceRule::Marginalise);
    let r = correlation(&out.variance, &truth);
    assert!(r > 0.7, "variance correlation was {r}");
}

#[test]
fn test_recovers_mean_log_quotients() {
    let (out, _, variance, mean) = fixture(3, VarianceRule::Marginalise);
    // The method estimates `ln alpha_g`, the *geometric* mean quotient. The
    // simulator's recipe (SPEC section 10, step 3) parametrises by the
    // arithmetic one, and the two differ by `v / 2`.
    let truth: Vec<f64> = mean
        .iter()
        .zip(&variance)
        .map(|(m, v)| m - 0.5 * v)
        .collect();
    let r = correlation(&out.mean_log_quotient, &truth);
    assert!(r > 0.98, "mean quotient correlation was {r}");
}

#[test]
fn test_error_bars_are_calibrated() {
    let (out, truth, _, _) = fixture(4, VarianceRule::Marginalise);
    // The residual in units of the reported error bar should have a standard
    // deviation near one if the error bars mean what they say.
    let z: Vec<f64> = out
        .log_fold_changes
        .iter()
        .zip(&truth)
        .zip(&out.error_bars)
        .map(|((est, act), e)| (est - act) / e)
        .collect();
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let sd = (z.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    assert!(
        sd > 0.5 && sd < 2.0,
        "standardised residual spread was {sd}"
    );
}

#[test]
fn test_point_estimate_rules_track_the_marginalising_one() {
    let (full, _, _, _) = fixture(5, VarianceRule::Marginalise);
    for rule in [VarianceRule::PosteriorMean, VarianceRule::MaxPosterior] {
        let (point, _, _, _) = fixture(5, rule);
        let r = correlation(&point.log_fold_changes, &full.log_fold_changes);
        assert!(r > 0.95, "{rule:?} correlated {r} with the full rule");
    }
}

#[test]
fn test_f32_storage_agrees_with_f64() {
    let sim = simulate(Some(SimulationParams {
        n_genes: 50,
        n_cells: 80,
        seed: 6,
        ..Default::default()
    }))
    .expect("simulates");

    let wide = sanity::<f64>(&sim.counts, &sim.cell_totals, None).expect("runs");
    let narrow = sanity::<f32>(&sim.counts, &sim.cell_totals, None).expect("runs");

    let pairs: [(&[f64], &[f32], &str); 5] = [
        (
            &wide.log_fold_changes,
            &narrow.log_fold_changes,
            "log_fold_changes",
        ),
        (&wide.error_bars, &narrow.error_bars, "error_bars"),
        (
            &wide.mean_log_quotient,
            &narrow.mean_log_quotient,
            "mean_log_quotient",
        ),
        (
            &wide.mean_log_quotient_error,
            &narrow.mean_log_quotient_error,
            "mean_log_quotient_error",
        ),
        (&wide.variance, &narrow.variance, "variance"),
    ];
    for (a, b, name) in pairs {
        assert_eq!(a.len(), b.len(), "{name} lengths differ");
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x - *y as f64).abs() < 1e-5 * x.abs().max(1.0),
                "{name}: {x} against {y}"
            );
        }
    }
}

#[test]
fn test_fixed_variance_runs_end_to_end() {
    let sim = simulate(Some(SimulationParams {
        n_genes: 40,
        n_cells: 60,
        seed: 11,
        ..Default::default()
    }))
    .expect("simulates");

    let params = SanityParams {
        variance_rule: VarianceRule::Fixed(0.8),
        ..Default::default()
    };
    let out = sanity::<f64>(&sim.counts, &sim.cell_totals, Some(params)).expect("runs");

    assert!(out.variance.iter().all(|&v| v == 0.8));
    assert!(out.log_fold_changes.iter().all(|x| x.is_finite()));
    assert!(out.error_bars.iter().all(|&e| e > 0.0 && e.is_finite()));
    assert!(out.mean_log_quotient.iter().all(|x| x.is_finite()));
}

#[test]
fn test_rejects_a_non_positive_fixed_variance() {
    use sanity_sc_rs::errors::SanityErrors;
    use sanity_sc_rs::input::CountMatrix;

    let counts = CountMatrix::new(vec![0, 2], vec![3, 5], vec![0, 2], 4).expect("well formed");
    let params = SanityParams {
        variance_rule: VarianceRule::Fixed(0.0),
        ..Default::default()
    };
    let err = sanity::<f64>(&counts, &[100.0, 200.0, 300.0, 50.0], Some(params)).unwrap_err();
    assert!(matches!(
        err,
        SanityErrors::InvalidFixedVariance { variance: 0.0 }
    ));
}

#[test]
fn test_log_transcription_quotients_add_the_gene_mean() {
    let sim = simulate(Some(SimulationParams {
        n_genes: 20,
        n_cells: 30,
        seed: 12,
        ..Default::default()
    }))
    .expect("simulates");

    let out = sanity::<f64>(&sim.counts, &sim.cell_totals, None).expect("runs");
    let ltq = out.log_transcription_quotients();

    assert_eq!(ltq.len(), out.n_genes * out.n_cells);
    for g in 0..out.n_genes {
        for c in 0..out.n_cells {
            let i = g * out.n_cells + c;
            let expected = out.log_fold_changes[i] + out.mean_log_quotient[g];
            assert!((ltq[i] - expected).abs() < 1e-12, "gene {g} cell {c}");
        }
    }
}

#[test]
fn test_single_umi_gene_stays_finite() {
    use sanity_sc_rs::input::CountMatrix;

    // A single UMI in a single cell carries no information about the variance,
    // so the gene should come back with wide error bars rather than a NaN or a
    // solver failure.
    let counts = CountMatrix::new(vec![2], vec![1], vec![0, 1], 4).expect("well formed");
    let out = sanity::<f64>(&counts, &[100.0, 200.0, 300.0, 50.0], None).expect("runs");

    assert!(out.log_fold_changes.iter().all(|x| x.is_finite()));
    assert!(out.error_bars.iter().all(|x| *x > 0.0 && x.is_finite()));
    assert!(out.mean_log_quotient.iter().all(|x| x.is_finite()));
    assert!(out.mean_log_quotient_error.iter().all(|x| x.is_finite()));

    // With nothing to go on the variance posterior stays near the prior, which
    // over a log-uniform grid to 50 has a mean well above one.
    assert!(
        out.variance[0] > 1.0,
        "single UMI gene variance {}",
        out.variance[0]
    );
}

#[test]
fn test_rejects_a_gene_with_no_counts() {
    use sanity_sc_rs::errors::SanityErrors;
    use sanity_sc_rs::input::CountMatrix;

    let counts = CountMatrix::new(vec![2], vec![5], vec![0, 1, 1], 4).expect("well formed");
    let err = sanity::<f64>(&counts, &[100.0, 200.0, 300.0, 50.0], None).unwrap_err();
    assert!(matches!(err, SanityErrors::EmptyGene { gene: 1 }));
}

#[test]
fn test_rejects_a_cell_with_no_library() {
    use sanity_sc_rs::errors::SanityErrors;
    use sanity_sc_rs::input::CountMatrix;

    let counts = CountMatrix::new(vec![0, 2], vec![3, 5], vec![0, 2], 4).expect("well formed");
    let err = sanity::<f64>(&counts, &[100.0, 0.0, 300.0, 50.0], None).unwrap_err();
    assert!(matches!(
        err,
        SanityErrors::NonPositiveCellTotal { index: 1, .. }
    ));
}
