//! CPU against GPU, every variance rule: wall clock and agreement.
//!
//! Positional arguments: n_genes n_cells n_bins seed library_size [rule]
//! [verbosity], where `rule` is one of `marginalise`, `mean`, `max`, `fixed`;
//! all four if absent. `verbosity` is 0, 1 or 2.
//!
//! cargo run --release --features gpu --example profile_gpu -- 500 20000

use std::time::Instant;

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use sanity_sc_rs::config::{DEFAULT_VARIANCE_BINS, SanityParams, VarianceRule, Verbosity};
use sanity_sc_rs::gpu::sanity_gpu;
use sanity_sc_rs::sanity;
use sanity_sc_rs::simulate::{SimulationParams, simulate};

/// Median, 99th percentile and maximum of a sample.
fn quantiles(mut x: Vec<f64>) -> (f64, f64, f64) {
    x.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let at = |q: f64| x[((x.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.99), x[x.len() - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: usize| -> usize {
        args.get(i)
            .map(|a| a.parse().expect("positional arguments are integers"))
            .unwrap_or(default)
    };
    let n_genes = arg(0, 500);
    let n_cells = arg(1, 20_000);
    let n_bins = arg(2, DEFAULT_VARIANCE_BINS);
    let seed = arg(3, 0) as u64;
    let library_size = arg(4, 500) as f64;
    let verbosity = match arg(6, 0) {
        0 => Verbosity::Quiet,
        1 => Verbosity::Normal,
        _ => Verbosity::Detailed,
    };

    let sim = simulate(Some(SimulationParams {
        n_genes,
        n_cells,
        seed,
        library_size,
        ..SimulationParams::default()
    }))
    .expect("the simulation parameters are valid");
    let n_genes = sim.counts.n_genes();
    println!("{n_genes} genes x {n_cells} cells, {n_bins} bins, library {library_size}");

    let client = WgpuRuntime::client(&WgpuDevice::default());
    // Compile both kernels before anything is timed.
    let warm = SanityParams::new(
        VarianceRule::Marginalise,
        1e-3,
        50.0,
        n_bins,
        Verbosity::Quiet,
    );
    sanity_gpu::<f64, WgpuRuntime>(&sim.counts, &sim.cell_totals, Some(warm), &client)
        .expect("warm-up run");

    for rule in [
        VarianceRule::Marginalise,
        VarianceRule::PosteriorMean,
        VarianceRule::MaxPosterior,
        VarianceRule::Fixed(1.0),
    ]
    .into_iter()
    .filter(|rule| match args.get(5).map(String::as_str) {
        None => true,
        Some("marginalise") => matches!(rule, VarianceRule::Marginalise),
        Some("mean") => matches!(rule, VarianceRule::PosteriorMean),
        Some("max") => matches!(rule, VarianceRule::MaxPosterior),
        Some("fixed") => matches!(rule, VarianceRule::Fixed(_)),
        Some(other) => panic!("unknown rule {other}"),
    }) {
        let params = SanityParams::new(rule, 1e-3, 50.0, n_bins, verbosity);
        let start = Instant::now();
        let cpu = sanity::<f64>(&sim.counts, &sim.cell_totals, Some(params)).expect("CPU run");
        let t_cpu = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let gpu =
            sanity_gpu::<f64, WgpuRuntime>(&sim.counts, &sim.cell_totals, Some(params), &client)
                .expect("GPU run");
        let t_gpu = start.elapsed().as_secs_f64();

        // A GPU run that did no work returns zeros, which no real error bar is.
        let dead = gpu
            .error_bars
            .iter()
            .filter(|&&e| e.is_nan() || e <= 0.0)
            .count();
        assert!(dead == 0, "GPU left {dead} error bars at zero or NaN");

        let mut fold = Vec::with_capacity(n_genes);
        let mut err = Vec::with_capacity(n_genes);
        let mut ltq = Vec::with_capacity(n_genes);
        for g in 0..n_genes {
            let (mut f, mut e, mut q) = (0.0f64, 0.0f64, 0.0f64);
            for i in g * n_cells..(g + 1) * n_cells {
                let ec = cpu.error_bars[i];
                f = f.max((gpu.log_fold_changes[i] - cpu.log_fold_changes[i]).abs() / ec);
                e = e.max((gpu.error_bars[i] - ec).abs() / ec);
                q = q.max(
                    (gpu.log_fold_changes[i] + gpu.mean_log_quotient[g]
                        - cpu.log_fold_changes[i]
                        - cpu.mean_log_quotient[g])
                        .abs(),
                );
            }
            fold.push(f);
            err.push(e);
            ltq.push(q);
        }
        let mean: Vec<f64> = (0..n_genes)
            .map(|g| {
                (gpu.mean_log_quotient[g] - cpu.mean_log_quotient[g]).abs()
                    / cpu.mean_log_quotient_error[g]
            })
            .collect();
        let var: Vec<f64> = (0..n_genes)
            .map(|g| (gpu.variance[g] - cpu.variance[g]).abs() / cpu.variance[g])
            .collect();

        println!(
            "\n{rule:?}: CPU {t_cpu:.3} s, GPU {t_gpu:.3} s, {:.1}x",
            t_cpu / t_gpu
        );
        let mut order: Vec<usize> = (0..n_genes).collect();
        order.sort_by(|&a, &b| fold[b].partial_cmp(&fold[a]).expect("no NaN"));
        for &g in order.iter().take(5) {
            let (_, values) = sim.counts.gene(g);
            let k: u64 = values.iter().map(|&x| x as u64).sum();
            println!(
                "  gene {g:4}: K {k:9}, nnz {:6}, v {:.3}, d err {:.2e}, m err {:.2e}",
                values.len(),
                cpu.variance[g],
                fold[g],
                mean[g]
            );
        }
        println!("  per-gene worst, median / 99% / max");
        for (label, x) in [
            ("d_c err / e_c", fold),
            ("e_c rel err", err),
            ("abs err m + d_c", ltq),
            ("m err / dm", mean),
            ("variance rel", var),
        ] {
            let (a, b, c) = quantiles(x);
            println!("  {label:16} {a:9.2e} {b:9.2e} {c:9.2e}");
        }
    }
}
