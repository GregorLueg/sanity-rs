//! Wall-clock driver for profiling a Sanity run.
//!
//! Builds a simulated matrix and runs every variance rule over it. Intended to
//! be run under a sampling profiler:
//!
//! ```text
//! cargo build --release --example profile_sanity
//! samply record -- ./target/release/examples/profile_sanity 2000 20000
//! ```
//!
//! Arguments are `n_genes`, `n_cells`, `n_bins`, `seed`, `library_size` and
//! `verbosity` (0, 1 or 2), all optional and positional.
//!
//! The default library size tracks the gene count at a quarter of a UMI per
//! gene per cell, which is the ratio a droplet experiment has (20 000 genes,
//! 5000 UMIs) and puts the matrix at the sparsity the kernels actually meet.
//! `SimulationParams::default` instead fixes 5000 UMIs regardless of the gene
//! count, which at any profiling size gives a matrix far denser than real data.

use std::time::Instant;

use sanity_sc_rs::config::{DEFAULT_VARIANCE_BINS, SanityParams, VarianceRule, Verbosity};
use sanity_sc_rs::sanity;
use sanity_sc_rs::simulate::{SimulationParams, simulate};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: usize| -> usize {
        args.get(i)
            .map(|a| a.parse().expect("positional arguments are integers"))
            .unwrap_or(default)
    };
    let n_genes = arg(0, 2000);
    let n_cells = arg(1, 20_000);
    let n_bins = arg(2, DEFAULT_VARIANCE_BINS);
    let seed = arg(3, 0) as u64;
    let library_size = arg(4, n_genes / 4) as f64;
    let verbosity = match arg(5, 0) {
        0 => Verbosity::Quiet,
        1 => Verbosity::Normal,
        _ => Verbosity::Detailed,
    };

    let sim_params = SimulationParams {
        n_genes,
        n_cells,
        seed,
        library_size,
        ..SimulationParams::default()
    };

    let start = Instant::now();
    let sim = simulate(Some(sim_params)).expect("the simulation parameters are valid");
    let n_genes = sim.counts.n_genes();
    let nnz: usize = (0..n_genes).map(|g| sim.counts.gene(g).0.len()).sum();
    println!(
        "simulated {n_genes} genes x {n_cells} cells, {nnz} stored counts ({:.1}% dense), {:.2} s",
        100.0 * nnz as f64 / (n_genes * n_cells) as f64,
        start.elapsed().as_secs_f64()
    );

    for rule in [
        VarianceRule::Marginalise,
        VarianceRule::PosteriorMean,
        VarianceRule::MaxPosterior,
        VarianceRule::Fixed(1.0),
    ] {
        let params = SanityParams::new(rule, 1e-3, 50.0, n_bins, verbosity);
        let start = Instant::now();
        let out = sanity::<f64>(&sim.counts, &sim.cell_totals, Some(params))
            .expect("the simulated matrix is well formed");
        let elapsed = start.elapsed().as_secs_f64();
        // Read an output back so the run cannot be optimised away.
        let checksum: f64 = out.variance.iter().sum();
        println!("{rule:?}: {elapsed:.3} s, sum of variances {checksum:.6}");
    }
}
