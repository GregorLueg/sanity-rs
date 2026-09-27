//! Output-level comparison against the numpy oracle.
//!
//! The fixture is produced by `reference/sanity_ref.py`, which transcribes
//! `docs/SPEC.md` directly and, where the spec rewrites an SI expression for
//! numerical reasons, uses the SI form instead. Agreement therefore checks the
//! rewrites as well as the code.
//!
//! Regenerate with:
//!
//! ```text
//! uv run --with numpy reference/sanity_ref.py --genes 40 --cells 60 \
//!     --v-bins 64 --out tests/data/oracle.txt
//! ```

use sanity_sc_rs::config::{SanityParams, VarianceRule, Verbosity};
use sanity_sc_rs::input::CountMatrix;
use sanity_sc_rs::sanity;

/// The parsed fixture.
struct Oracle {
    n_genes: usize,
    n_cells: usize,
    grid: (f64, f64, usize),
    totals: Vec<f64>,
    indices: Vec<u32>,
    values: Vec<u32>,
    indptr: Vec<usize>,
    fold: Vec<f64>,
    error: Vec<f64>,
    mean: Vec<f64>,
    mean_error: Vec<f64>,
    variance: Vec<f64>,
}

/// Read the whitespace-keyed fixture format the oracle writes.
fn load(path: &str) -> Oracle {
    let text = std::fs::read_to_string(path).expect("the oracle fixture is committed");
    let mut o = Oracle {
        n_genes: 0,
        n_cells: 0,
        grid: (0.0, 0.0, 0),
        totals: Vec::new(),
        indices: Vec::new(),
        values: Vec::new(),
        indptr: vec![0],
        fold: Vec::new(),
        error: Vec::new(),
        mean: Vec::new(),
        mean_error: Vec::new(),
        variance: Vec::new(),
    };

    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else { continue };
        let rest: Vec<&str> = parts.collect();
        match key {
            "n_genes" => o.n_genes = rest[0].parse().unwrap(),
            "n_cells" => o.n_cells = rest[0].parse().unwrap(),
            "grid" => {
                o.grid = (
                    rest[0].parse().unwrap(),
                    rest[1].parse().unwrap(),
                    rest[2].parse().unwrap(),
                )
            }
            "totals" => o.totals = rest.iter().map(|x| x.parse().unwrap()).collect(),
            "indices" => o
                .indices
                .extend(rest.iter().map(|x| x.parse::<u32>().unwrap())),
            "values" => {
                o.values
                    .extend(rest.iter().map(|x| x.parse::<u32>().unwrap()));
                o.indptr.push(o.indices.len());
            }
            "summary" => {
                o.mean.push(rest[0].parse().unwrap());
                o.mean_error.push(rest[1].parse().unwrap());
                o.variance.push(rest[2].parse().unwrap());
            }
            "fold" => o
                .fold
                .extend(rest.iter().map(|x| x.parse::<f64>().unwrap())),
            "error" => o
                .error
                .extend(rest.iter().map(|x| x.parse::<f64>().unwrap())),
            _ => {}
        }
    }
    o
}

/// Largest relative discrepancy between two slices, on a floor of `scale`.
fn max_relative(a: &[f64], b: &[f64], scale: f64) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(scale))
        .fold(0.0f64, f64::max)
}

#[test]
fn test_matches_the_numpy_oracle() {
    let o = load("tests/data/oracle.txt");
    let counts = CountMatrix::new(
        o.indices.clone(),
        o.values.clone(),
        o.indptr.clone(),
        o.n_cells,
    )
    .expect("the fixture is well formed");

    let params = SanityParams::new(
        VarianceRule::Marginalise,
        o.grid.0,
        o.grid.1,
        o.grid.2,
        Verbosity::Quiet,
    );
    let out = sanity::<f64>(&counts, &o.totals, Some(params)).expect("the run succeeds");

    assert_eq!(out.n_genes, o.n_genes);
    assert_eq!(out.n_cells, o.n_cells);

    // The oracle bisects where the crate runs Newton, and uses the SI's
    // cancellation-prone determinant where the crate uses the rewritten one, so
    // the two agree to solver tolerance rather than to the last bit.
    assert!(
        max_relative(&out.log_fold_changes, &o.fold, 1e-6) < 1e-7,
        "log fold changes: {:e}",
        max_relative(&out.log_fold_changes, &o.fold, 1e-6)
    );
    assert!(
        max_relative(&out.error_bars, &o.error, 1e-6) < 1e-7,
        "error bars: {:e}",
        max_relative(&out.error_bars, &o.error, 1e-6)
    );
    assert!(
        max_relative(&out.mean_log_quotient, &o.mean, 1e-6) < 1e-7,
        "mean log quotients: {:e}",
        max_relative(&out.mean_log_quotient, &o.mean, 1e-6)
    );
    assert!(
        max_relative(&out.mean_log_quotient_error, &o.mean_error, 1e-6) < 1e-7,
        "mean error bars: {:e}",
        max_relative(&out.mean_log_quotient_error, &o.mean_error, 1e-6)
    );
    assert!(
        max_relative(&out.variance, &o.variance, 1e-6) < 1e-7,
        "variances: {:e}",
        max_relative(&out.variance, &o.variance, 1e-6)
    );
}
