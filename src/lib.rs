//! Bayesian inference of gene expression states from single cell UMI counts.
//!
//! A clean-room implementation of the Sanity method. Given raw UMI counts and
//! per-cell totals, this crate returns a posterior log expression level and an
//! error bar for every gene in every cell.
//!
//! Genes are independent under this model, so the whole method is one parallel
//! iteration over genes. A gene's input is its sparse column plus the full
//! vector of per-cell totals, because a cell with zero counts for that gene
//! still contributes through its library size.
//!
//! Inputs are raw UMI counts. Anything already log-normalised is the wrong
//! input: the point of the method is to model the Poisson sampling in the raw
//! counts.
//!
//! # Provenance
//!
//! The reference implementation (`jmbreda/Sanity`) is GPL-3.0 and is **not** a
//! source for this crate. Everything here is written from `docs/SPEC.md`, which
//! restates the mathematics of the paper's Supplementary Information. See
//! `docs/PROVENANCE.md` for the full position.
//!
//! # References
//!
//! Breda, Zavolan, van Nimwegen. *Bayesian inference of gene expression states
//! from single-cell RNA-seq data.* Nature Biotechnology 39(8):1008-1016 (2021).

#![warn(missing_docs)]

pub mod config;
pub mod errors;
pub mod float;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod input;
pub mod simulate;

mod model;
mod utils;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::config::{SanityParams, VarianceGrid, VarianceRule};
use crate::errors::SanityErrors;
use crate::float::{SanityFloat, narrow};
use crate::input::CountMatrix;
use crate::model::empty_cells::ShiftTable;
use crate::model::gene::{GeneScratch, GeneSummary, empty_cell_fit, run_gene};
use crate::utils::progress::report_decile_progress;

//////////////////
// SanityOutput //
//////////////////

/// Everything a run produces.
///
/// The two per-cell matrices are gene-major and dense: output row `g`, cell `c`
/// is at `g * n_cells + c`. Row `g` is input gene `genes[g]`.
#[derive(Clone, Debug)]
pub struct SanityOutput<T: SanityFloat> {
    /// Posterior log fold change of each gene in each cell, `d_c`.
    pub log_fold_changes: Vec<T>,
    /// Error bar on each log fold change, `e_c`.
    pub error_bars: Vec<T>,
    /// Posterior mean log transcription quotient of each gene, `m`.
    ///
    /// The *geometric* mean: the prior centres the log fold changes on zero, so
    /// the arithmetic mean quotient is `exp(m + v / 2)`.
    pub mean_log_quotient: Vec<T>,
    /// Error bar on each `m`.
    pub mean_log_quotient_error: Vec<T>,
    /// Posterior estimate of each gene's variance in log fold change.
    pub variance: Vec<T>,
    /// Input index of each output row, ascending. `0..n_genes` for [`sanity`],
    /// the kept genes for [`sanity_select`].
    pub genes: Vec<usize>,
    /// Number of genes in the output, `genes.len()`.
    pub n_genes: usize,
    /// Number of cells.
    pub n_cells: usize,
}

impl<T: SanityFloat> SanityOutput<T> {
    /// The normalised expression matrix, `m + d_c`.
    ///
    /// This is the log transcription quotient: the quantity downstream
    /// consumers read as the normalised expression level. Its error bar is
    /// [`SanityOutput::error_bars`]; [`SanityOutput::mean_log_quotient_error`]
    /// is the separate uncertainty on the gene's overall level.
    ///
    /// ### Returns
    ///
    /// A freshly allocated `n_genes * n_cells` gene-major matrix.
    pub fn log_transcription_quotients(&self) -> Vec<T> {
        let mut out = self.log_fold_changes.clone();
        for (g, row) in out.chunks_exact_mut(self.n_cells).enumerate() {
            let mean = self.mean_log_quotient[g];
            for x in row.iter_mut() {
                *x = *x + mean;
            }
        }
        out
    }
}

/// Run Sanity over a count matrix.
///
/// One `par_iter` over genes; there is no other loop. Each Rayon worker holds
/// scratch of `O(n_cells + n_bins)` and reuses it across the genes it is given.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes, not just
///   those present in `counts`. Taken as `f64` because these are integer counts
///   in disguise and are exact there to `2^53`, which `f32` is not.
/// * `params` - Run parameters, or [`SanityParams::default`].
///
/// ### Returns
///
/// The posterior log fold changes, their error bars, and the per-gene means,
/// error bars and variances.
pub fn sanity<T: SanityFloat>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
) -> Result<SanityOutput<T>, SanityErrors> {
    let params = params.unwrap_or_default();
    let n_cells = counts.n_cells();
    let n_genes = counts.n_genes();
    let (grid, log_totals, log_total_sum) = prepare_run(counts, cell_totals, &params)?;
    let table = shift_table(counts, &log_totals, &params);
    let fit = empty_cell_fit(&log_totals);

    let verbose = params.verbosity.normal_verbosity();
    if verbose {
        print_run_header("CPU", n_genes, n_cells, &params);
    }
    let start = Instant::now();
    let done = AtomicUsize::new(0);

    let mut log_fold_changes = vec![T::zero(); n_genes * n_cells];
    let mut error_bars = vec![T::zero(); n_genes * n_cells];
    let mut mean_log_quotient = vec![T::zero(); n_genes];
    let mut mean_log_quotient_error = vec![T::zero(); n_genes];
    let mut variance = vec![T::zero(); n_genes];

    let n_bins = grid.len();
    (
        log_fold_changes.par_chunks_mut(n_cells),
        error_bars.par_chunks_mut(n_cells),
        mean_log_quotient.par_iter_mut(),
        mean_log_quotient_error.par_iter_mut(),
        variance.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each_init(
            || {
                (
                    GeneScratch::new(n_cells, n_bins),
                    vec![0.0f64; n_cells],
                    vec![0.0f64; n_cells],
                )
            },
            |(scratch, row_d, row_e), (gene, (out_d, out_e, out_m, out_dm, out_v))| {
                let (indices, values) = counts.gene(gene);
                let summary = run_gene(
                    indices,
                    values,
                    &log_totals,
                    log_total_sum,
                    &grid,
                    table.as_ref(),
                    &fit,
                    &params,
                    scratch,
                    row_d,
                    row_e,
                )?;

                for c in 0..n_cells {
                    out_d[c] = narrow(row_d[c]);
                    out_e[c] = narrow(row_e[c]);
                }
                *out_m = narrow(summary.mean_log_quotient);
                *out_dm = narrow(summary.mean_log_quotient_error);
                *out_v = narrow(summary.variance);

                if verbose {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    report_decile_progress(d, d - 1, n_genes, "genes", start.elapsed());
                }
                Ok::<(), SanityErrors>(())
            },
        )?;

    Ok(SanityOutput {
        log_fold_changes,
        error_bars,
        mean_log_quotient,
        mean_log_quotient_error,
        variance,
        genes: (0..n_genes).collect(),
        n_genes,
        n_cells,
    })
}

/// Print the one-line summary a verbose run opens with.
///
/// ### Params
///
/// * `device` - Where the run happens, `"CPU"` or `"GPU"`.
/// * `n_genes` - Genes in the input.
/// * `n_cells` - Cells in the input.
/// * `params` - Resolved run parameters.
pub(crate) fn print_run_header(
    device: &str,
    n_genes: usize,
    n_cells: usize,
    params: &SanityParams,
) {
    let grid = if params.variance_rule.needs_grid() {
        format!(", {} variance bins", params.variance_bins)
    } else {
        String::new()
    };
    println!(
        "Sanity ({device}): {n_genes} genes x {n_cells} cells, {:?}{grid}",
        params.variance_rule
    );
}

/// Validate the run inputs and derive what every gene shares.
///
/// `ln(sum_c T_c)` is the only quantity in the method that spans genes, and it
/// comes from the totals alone. That is what lets [`sanity_select`] decide a
/// gene's fate inside the gene loop.
///
/// ### Params
///
/// * `counts` - The count matrix; checked for genes with no counts.
/// * `cell_totals` - Total UMI count of every cell.
/// * `params` - Resolved run parameters.
///
/// ### Returns
///
/// The variance grid, `ln T_c` per cell and `ln(sum_c T_c)`, or the first
/// input error found.
pub(crate) fn prepare_run(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: &SanityParams,
) -> Result<(VarianceGrid, Vec<f64>, f64), SanityErrors> {
    let n_cells = counts.n_cells();
    if let Some(gene) = (0..counts.n_genes()).find(|&g| counts.gene(g).1.iter().all(|&k| k == 0)) {
        return Err(SanityErrors::EmptyGene { gene });
    }
    if cell_totals.len() != n_cells {
        return Err(SanityErrors::CellTotalsLengthMismatch {
            n_totals: cell_totals.len(),
            n_cells,
        });
    }
    if let Some((index, total)) = cell_totals
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, t)| !t.is_finite() || t <= 0.0)
    {
        return Err(SanityErrors::NonPositiveCellTotal { index, total });
    }
    if let VarianceRule::Fixed(v) = params.variance_rule
        && (!v.is_finite() || v <= 0.0)
    {
        return Err(SanityErrors::InvalidFixedVariance { variance: v });
    }
    if params.variance_rule.needs_grid()
        && (!params.variance_min.is_finite()
            || params.variance_min <= 0.0
            || !params.variance_max.is_finite()
            || params.variance_max <= params.variance_min
            || params.variance_bins == 0)
    {
        return Err(SanityErrors::InvalidVarianceGrid {
            min: params.variance_min,
            max: params.variance_max,
            bins: params.variance_bins,
        });
    }

    let grid = VarianceGrid::new(
        params.variance_min,
        params.variance_max,
        params.variance_bins,
    );
    let log_totals: Vec<f64> = cell_totals.iter().map(|t| t.ln()).collect();
    let log_total_sum = cell_totals.iter().sum::<f64>().ln();

    Ok((grid, log_totals, log_total_sum))
}

/// Build the table of empty-cell sums, if the rule sweeps a grid.
///
/// `VarianceRule::Fixed` solves once per gene, where the build would cost more
/// than it saves.
///
/// ### Params
///
/// * `counts` - The count matrix, for the largest `K`.
/// * `log_totals` - `ln T_c` for every cell.
/// * `params` - Run parameters.
///
/// ### Returns
///
/// The table, or `None` for a rule without a grid.
pub(crate) fn shift_table(
    counts: &CountMatrix,
    log_totals: &[f64],
    params: &SanityParams,
) -> Option<ShiftTable> {
    if !params.variance_rule.needs_grid() {
        return None;
    }
    let max_k = (0..counts.n_genes())
        .map(|g| counts.gene(g).1.iter().map(|&k| k as f64).sum::<f64>())
        .fold(0.0, f64::max);
    Some(ShiftTable::new(log_totals, params.variance_max * max_k))
}

////////////////////
// Gene selection //
////////////////////

/// One finished gene, as the keep predicate of [`sanity_select`] sees it.
///
/// Everything is `f64` whatever the storage type, because the predicate runs
/// before narrowing.
#[derive(Clone, Copy, Debug)]
pub struct GeneView<'a> {
    /// Posterior log fold change in each cell, `d_c`, length `n_cells`.
    pub log_fold_changes: &'a [f64],
    /// Error bar on each log fold change, `e_c`, length `n_cells`.
    pub error_bars: &'a [f64],
    /// Posterior mean log transcription quotient, `m`.
    pub mean_log_quotient: f64,
    /// Error bar on `m`.
    pub mean_log_quotient_error: f64,
    /// Posterior estimate of the gene's variance in log fold change.
    pub variance: f64,
}

/// One kept gene: its input index, both rows narrowed to storage, and its
/// summary.
type KeptGene<T> = (usize, Vec<T>, Vec<T>, GeneSummary);

/// Run Sanity and store only the genes a predicate keeps.
///
/// Same inference as [`sanity`], gene for gene. The predicate is evaluated on
/// each finished gene inside the gene loop, so a rejected gene is never stored
/// and the output is dense over kept genes only. Resident output is
/// `2 * n_kept * n_cells` values instead of `2 * n_genes * n_cells`.
///
/// The predicate sees one gene at a time. A rule that needs every gene before
/// deciding, a quantile cut for instance, cannot be expressed here; run
/// [`sanity`] for that.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes, as for
///   [`sanity`].
/// * `params` - Run parameters, or [`SanityParams::default`].
/// * `keep` - Returns `true` for a gene to store. Called once per gene, from
///   any Rayon worker.
///
/// ### Returns
///
/// The kept genes in input order, with their input indices in
/// [`SanityOutput::genes`]. Keeping nothing is not an error; the output is then
/// empty.
pub fn sanity_select<T, F>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
    keep: F,
) -> Result<SanityOutput<T>, SanityErrors>
where
    T: SanityFloat,
    F: Fn(GeneView<'_>) -> bool + Sync,
{
    let params = params.unwrap_or_default();
    let n_cells = counts.n_cells();
    let n_genes = counts.n_genes();
    let (grid, log_totals, log_total_sum) = prepare_run(counts, cell_totals, &params)?;
    let table = shift_table(counts, &log_totals, &params);
    let fit = empty_cell_fit(&log_totals);

    let verbose = params.verbosity.normal_verbosity();
    if verbose {
        print_run_header("CPU", n_genes, n_cells, &params);
    }
    let start = Instant::now();
    let done = AtomicUsize::new(0);

    let n_bins = grid.len();
    let kept: Vec<Option<KeptGene<T>>> = (0..n_genes)
        .into_par_iter()
        .map_init(
            || {
                (
                    GeneScratch::new(n_cells, n_bins),
                    vec![0.0f64; n_cells],
                    vec![0.0f64; n_cells],
                )
            },
            |(scratch, row_d, row_e), gene| {
                let (indices, values) = counts.gene(gene);
                let summary = run_gene(
                    indices,
                    values,
                    &log_totals,
                    log_total_sum,
                    &grid,
                    table.as_ref(),
                    &fit,
                    &params,
                    scratch,
                    row_d,
                    row_e,
                )?;
                let view = GeneView {
                    log_fold_changes: row_d,
                    error_bars: row_e,
                    mean_log_quotient: summary.mean_log_quotient,
                    mean_log_quotient_error: summary.mean_log_quotient_error,
                    variance: summary.variance,
                };
                if verbose {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    report_decile_progress(d, d - 1, n_genes, "genes", start.elapsed());
                }
                if !keep(view) {
                    return Ok(None);
                }
                let d: Vec<T> = row_d.iter().map(|&x| narrow(x)).collect();
                let e: Vec<T> = row_e.iter().map(|&x| narrow(x)).collect();
                Ok(Some((gene, d, e, summary)))
            },
        )
        .collect::<Result<_, SanityErrors>>()?;

    let n_kept = kept.iter().flatten().count();
    if verbose {
        println!("  Kept {n_kept} / {n_genes} genes");
    }
    let mut out = SanityOutput {
        log_fold_changes: Vec::with_capacity(n_kept * n_cells),
        error_bars: Vec::with_capacity(n_kept * n_cells),
        mean_log_quotient: Vec::with_capacity(n_kept),
        mean_log_quotient_error: Vec::with_capacity(n_kept),
        variance: Vec::with_capacity(n_kept),
        genes: Vec::with_capacity(n_kept),
        n_genes: n_kept,
        n_cells,
    };

    for (gene, d, e, summary) in kept.into_iter().flatten() {
        out.log_fold_changes.extend_from_slice(&d);
        out.error_bars.extend_from_slice(&e);
        out.mean_log_quotient
            .push(narrow(summary.mean_log_quotient));
        out.mean_log_quotient_error
            .push(narrow(summary.mean_log_quotient_error));
        out.variance.push(narrow(summary.variance));
        out.genes.push(gene);
    }
    Ok(out)
}
