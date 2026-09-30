//! The CubeCL path. Feature-gated on `gpu`, CPU-only build otherwise.
//!
//! Everything on the device is `f32`, because wgpu exposes no `f64` on any
//! backend. That is a departure from the crate's numeric policy, and the reason
//! this path is held to its own measured tolerance rather than to the CPU's.
//!
//! ### Mapping
//!
//! The first pass is [`fn@kernels::sweep_grid_gpu`]: one workgroup of whole planes
//! per gene, running the whole variance grid, with each cell sweep reduced
//! across the workgroup so every thread agrees on the offset solve. A gene at
//! most `SPARSE_MAX_DENSITY` dense walks only its nonzero cells and takes the
//! empty cells' sums from the CPU's `ShiftTable`, uploaded in `f32` to steer
//! the solve; the host adds the same sums back in `f64` (see [`finish_sweep`]).
//! The host then assembles each bin's log marginal likelihood in `f64` and
//! forms the weights.
//! The second pass is [`fn@kernels::marginalise_gpu`]: one thread per gene and
//! cell, integrating over the kept bins with the offsets the first pass left
//! on the device.
//!
//! ### Precision
//!
//! Compensated summation is no defence on this backend: wgpu compiles Metal
//! shaders with fast-math on, which folds a `two_sum`. So the arithmetic is
//! arranged so that nothing cancels, by five exact rewrites:
//!
//! * **The offset term leaves the likelihood.** With `d_c = t_c - ln T_c -
//!   ln(v s) + z` and `s = sum_c k_c`, `sum_c k_c d_c - s z = sum_c k_c t_c -
//!   sum_c k_c ln T_c - s ln(v s)`. The pair `sum_c k_c d_c` and `s z`, each of
//!   order `1e6` for an expressed gene, never forms.
//! * **Every per-cell term is of order one.** The stationarity condition is
//!   `omega_c = v k_c - d_c`, so the offset residual `sum_c omega_c - v s` is
//!   `-sum_c d_c`, an empty cell's log fold change is `-omega_c`, and the two
//!   large likelihood terms, `k_c ln omega_c` and `ln(1 + omega_c)`, split into
//!   an anchor the host knows exactly plus `ln(1 - d_c / (v k_c))` and
//!   `ln(1 + (1 - d_c) / (v k_c))`. See `kernels::sweep`.
//! * **The host assembles the likelihood.** The device returns only the five
//!   sums; the combination, which does cancel, happens in `f64`.
//! * **The likelihood is taken where the solve stopped.** The `-s z` form holds
//!   only at an exact root; the host adds the term the residual leaves, which
//!   turns a first-order error into a second-order one. See `log_marginal`.
//! * **The offset is solved relative to an `f64` anchor.** `z` is of order 20,
//!   where `f32` resolves only `2e-6`, and the Laplace determinant is not
//!   stationary in it. See `kernels::OFFSET_ULPS_F32`.
//!
//! Without the second rewrite, the sums were of order `v s` and `K ln K`. On a
//! simulated gene holding a quarter of the reads (`K = 2.6e6`) that left its log
//! fold changes `1.6e-2` of an error bar from the CPU's, above the 1% to which
//! the grid itself is resolved ([`crate::config::DEFAULT_VARIANCE_BINS`]).
//! With all five, measured 2026-09-24 against the CPU path, the worst log fold
//! change is `1.9e-4` of an error bar at 2000 genes by 20000 cells and `2.7e-3`
//! at 500 by 200000; `docs/BENCHMARKS.md` has the full table.

pub mod kernels;

use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rayon::prelude::*;

use crate::config::{SanityParams, VarianceRule};
use crate::errors::SanityErrors;
use crate::float::{SanityFloat, narrow};
use crate::input::CountMatrix;
use crate::model::empty_cells::{EmptySums, N_SUMS, PANEL_POINTS, ShiftTable};
use crate::model::fractions::NonzeroCells;
use crate::model::gene::{
    MARGINALISE_MIN_WEIGHT, SPARSE_MAX_DENSITY, collapse_target, log_marginal_at,
    log_marginal_at_sparse, marginal_summary, posterior_weights,
};
use crate::utils::polygamma::{digamma, trigamma};
use crate::utils::progress::report_decile_progress;
use crate::{GeneView, SanityOutput, prepare_run, print_run_header, shift_table};

use self::kernels::{marginalise_gpu, sweep_grid_gpu};

////////////
// Consts //
////////////

/// Target cells per lane in the first pass, which sets how many planes share
/// one gene.
const CELLS_PER_LANE: usize = 160;

/// Preferred workgroup width of the second pass, which has no reduction and
/// needs no particular width.
const MARGINALISE_WORKGROUP: u32 = 256;

/// Log-likelihood gap, per cell, below which [`VarianceRule::MaxPosterior`]
/// treats a bin as tied with the device's best and settles it in `f64`.
const MAX_POSTERIOR_TIE_MARGIN_PER_CELL: f64 = 5e-7;

/// The tie margin for a run over `n_cells` cells.
///
/// ### Params
///
/// * `n_cells` - Cells.
///
/// ### Returns
///
/// [`MAX_POSTERIOR_TIE_MARGIN_PER_CELL`] times the cells.
fn tie_margin(n_cells: usize) -> f64 {
    MAX_POSTERIOR_TIE_MARGIN_PER_CELL * n_cells as f64
}

/// Ceiling on the largest buffer of one batch, the second pass's output.
///
/// Batching bounds device memory; the device's own per-binding limit applies on
/// top. At 20000 cells, 512 MiB is 3355 genes per batch.
const GPU_BATCH_BYTES: u64 = 512 << 20;

/////////////
// Staging //
/////////////

/// One batch's resident inputs.
struct Batch<R: Runtime> {
    /// Dense counts, `[gene * n_cells + c]`, for the second pass.
    counts: GpuTensor<R, f32>,
    /// Start of each gene's listed cells for the first pass, length
    /// `n_genes + 1`. Only table genes list any.
    list_offsets: GpuTensor<R, u32>,
    /// Cell index of each listed cell.
    list_cells: GpuTensor<R, u32>,
    /// Count of each listed cell.
    list_counts: GpuTensor<R, f32>,
    /// Most cells any one gene of the batch walks in the first pass.
    max_walked: usize,
    /// Cells per dense row.
    counts_cells: usize,
    /// Per gene, whether its empty cells come from the table, in which case
    /// only its nonzero cells are listed.
    use_table: Vec<bool>,
    /// `K` per gene.
    totals: Vec<f64>,
    /// Cells with a non-zero count, per gene.
    n_expressed: Vec<usize>,
    /// Input index of the first gene.
    first: usize,
    /// Genes in the batch.
    n_genes: usize,
}

/// The empty-cell table on the device, or placeholders when the run has none.
struct DeviceTable<R: Runtime> {
    /// Panel boundaries.
    breaks: GpuTensor<R, f32>,
    /// Midpoint anchors.
    anchors: GpuTensor<R, f32>,
    /// Chebyshev coefficients.
    coeffs: GpuTensor<R, f32>,
    /// `ln(sum_c T_c)` and `ln(sum_c T_c^2)`.
    scalars: GpuTensor<R, f32>,
    /// Number of panels.
    n_panels: u32,
}

impl<R: Runtime> DeviceTable<R> {
    /// Upload a table, or a one-panel placeholder that no gene reads.
    ///
    /// ### Params
    ///
    /// * `table` - The run's table, if it has one.
    /// * `client` - CubeCL compute client.
    ///
    /// ### Returns
    ///
    /// The device table, or an upload error.
    fn new(table: Option<&ShiftTable>, client: &ComputeClient<R>) -> Result<Self, SanityErrors> {
        let [breaks, anchors, coeffs, scalars] = match table {
            Some(t) => t.to_f32(),
            None => [
                vec![0.0, 1.0],
                vec![1.0; N_SUMS],
                vec![0.0; N_SUMS * PANEL_POINTS],
                vec![0.0; 2],
            ],
        };
        let n_panels = (breaks.len() - 1) as u32;
        let up = |x: &[f32]| GpuTensor::<R, f32>::from_slice(x, vec![x.len()], client);
        Ok(Self {
            breaks: up(&breaks)?,
            anchors: up(&anchors)?,
            coeffs: up(&coeffs)?,
            scalars: up(&scalars)?,
            n_panels,
        })
    }
}

/// The first pass's output, on the device and read back.
struct SweepResult<R: Runtime> {
    /// The raw output, left on the device for the second pass.
    device: GpuTensor<R, f32>,
    /// Per-gene, per-bin parameters the pass ran with; see
    /// [`fn@kernels::sweep_grid_gpu`].
    bins: GpuTensor<R, f32>,
    /// Per-gene scalars the pass ran with.
    scalars: GpuTensor<R, f32>,
    /// The output read back in `f64`, `[(r * n_genes + gene) * n_bins + b]`,
    /// with each table gene's empty-cell share added; see [`finish_sweep`].
    /// Row zero is the anchored offset; [`SweepResult::offsets`] has it
    /// unanchored.
    host: Vec<f64>,
    /// `z(v_b)` per gene and bin, the anchor added back in `f64`.
    offsets: Vec<f64>,
}

/// The host's anchor for the offset at one variance, `ln(sum_c T_c) + v / 2`.
///
/// The device solves for `z` relative to it; see [`fn@kernels::sweep_grid_gpu`].
/// It is the cold guess, so the anchored offset is of order one.
///
/// ### Params
///
/// * `log_total_sum` - `ln(sum_c T_c)`.
/// * `v` - The variance.
///
/// ### Returns
///
/// The anchor, in `f64`.
fn offset_anchor(log_total_sum: f64, v: f64) -> f64 {
    log_total_sum + 0.5 * v
}

/// Upload one batch of genes: dense rows, and the first pass's cell lists.
///
/// With a table, a gene at most [`SPARSE_MAX_DENSITY`] dense lists its nonzero
/// cells; every other gene lists none and walks its dense row.
///
/// ### Params
///
/// * `counts` - The count matrix.
/// * `first` - Input index of the first gene.
/// * `n_genes` - Genes in the batch.
/// * `has_table` - Whether the run has an empty-cell table.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The batch, or a device-limit error.
fn stage_batch<R: Runtime>(
    counts: &CountMatrix,
    first: usize,
    n_genes: usize,
    has_table: bool,
    client: &ComputeClient<R>,
) -> Result<Batch<R>, SanityErrors> {
    let n_cells = counts.n_cells();
    let mut dense = vec![0.0f32; n_genes * n_cells];
    dense
        .par_chunks_mut(n_cells)
        .enumerate()
        .for_each(|(g, row)| {
            let (indices, values) = counts.gene(first + g);
            for (&i, &k) in indices.iter().zip(values) {
                row[i as usize] = k as f32;
            }
        });
    let (totals, n_expressed) = (first..first + n_genes)
        .map(|g| {
            let values = counts.gene(g).1;
            (
                values.iter().map(|&x| x as f64).sum::<f64>(),
                values.iter().filter(|&&x| x > 0).count(),
            )
        })
        .unzip();

    let use_table: Vec<bool> = (first..first + n_genes)
        .map(|g| has_table && counts.gene(g).0.len() as f64 <= SPARSE_MAX_DENSITY * n_cells as f64)
        .collect();
    let mut list_offsets = Vec::with_capacity(n_genes + 1);
    let mut list_cells: Vec<u32> = Vec::new();
    let mut list_counts: Vec<f32> = Vec::new();
    list_offsets.push(0u32);
    let mut max_walked = 0;
    for (g, &sparse) in use_table.iter().enumerate() {
        if sparse {
            let (indices, values) = counts.gene(first + g);
            for (&i, &k) in indices.iter().zip(values) {
                if k > 0 {
                    list_cells.push(i);
                    list_counts.push(k as f32);
                }
            }
            max_walked =
                max_walked.max(list_cells.len() - *list_offsets.last().unwrap_or(&0) as usize);
        } else {
            max_walked = n_cells;
        }
        list_offsets.push(list_cells.len() as u32);
    }
    // A batch with nothing listed still needs non-empty bindings.
    if list_cells.is_empty() {
        list_cells.push(0);
        list_counts.push(0.0);
    }

    Ok(Batch {
        counts: GpuTensor::from_slice(&dense, vec![n_genes * n_cells], client)?,
        list_offsets: GpuTensor::from_slice(&list_offsets, vec![n_genes + 1], client)?,
        list_cells: GpuTensor::from_slice(&list_cells, vec![list_cells.len()], client)?,
        list_counts: GpuTensor::from_slice(&list_counts, vec![list_counts.len()], client)?,
        max_walked,
        counts_cells: n_cells,
        use_table,
        totals,
        n_expressed,
        first,
        n_genes,
    })
}

/// How many planes share one gene in the first pass.
///
/// Enough that each lane walks about [`CELLS_PER_LANE`] listed cells, rounded
/// up to a power of two, and no more than the device or the kernel's shared
/// scratch allows. Sized for the widest plane the device reports; a driver that
/// picks a narrower one gives each gene more planes, which the cap accounts
/// for.
///
/// ### Params
///
/// * `n_cells` - Most cells any gene of the batch walks.
/// * `limits` - The device's limits.
///
/// ### Returns
///
/// Planes per gene, or [`SanityErrors::GpuUnsupported`] on a device without
/// plane operations.
fn planes_per_gene(n_cells: usize, limits: &GpuLimits) -> Result<u32, SanityErrors> {
    let plane = limits.plane_size_max;
    if plane == 0 || plane > limits.max_units_per_cube {
        return Err(SanityErrors::GpuUnsupported {
            reason: format!(
                "the first pass needs plane operations; this device reports planes of {} to {} lanes and at most {} units per cube",
                limits.plane_size_min, plane, limits.max_units_per_cube
            ),
        });
    }
    // The kernel's shared scratch holds `MAX_PLANES_PER_GENE` planes, and a
    // workgroup sized for the widest plane holds more of the narrowest. Past
    // the scratch the writes would land out of bounds without an error.
    let narrowest = limits.plane_size_min.clamp(1, plane);
    let cap =
        (limits.max_units_per_cube / plane).min(kernels::MAX_PLANES_PER_GENE * narrowest / plane);
    if cap == 0 {
        return Err(SanityErrors::GpuUnsupported {
            reason: format!(
                "planes of {} to {} lanes do not fit the first pass's scratch of {} planes",
                limits.plane_size_min,
                plane,
                kernels::MAX_PLANES_PER_GENE
            ),
        });
    }
    let wanted = n_cells.div_ceil(CELLS_PER_LANE * plane as usize).max(1);
    Ok((wanted.next_power_of_two() as u32).min(cap))
}

/// Run the first pass over a batch.
///
/// ### Params
///
/// * `batch` - The resident batch.
/// * `log_totals` - `ln T_c` on the device.
/// * `table` - The run's empty-cell table, if it has one.
/// * `table_dev` - The same table on the device.
/// * `log_total_sum` - `ln(sum_c T_c)`, which fixes the offset anchors.
/// * `bin_v` - `v` per gene and bin, `[gene * n_bins + b]`.
/// * `guess` - First bin's offset guess per gene.
/// * `cold_start` - Whether the guess is cold.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The pass's output, or [`SanityErrors::GpuOffsetSolveDiverged`] for the first
/// gene whose solve failed.
#[allow(clippy::too_many_arguments)]
fn run_sweep<R: Runtime>(
    batch: &Batch<R>,
    log_totals: &GpuTensor<R, f32>,
    table: Option<&ShiftTable>,
    table_dev: &DeviceTable<R>,
    log_total_sum: f64,
    bin_v: &[f64],
    guess: &[f64],
    cold_start: bool,
    client: &ComputeClient<R>,
) -> Result<SweepResult<R>, SanityErrors> {
    let n_genes = batch.n_genes;
    let n_bins = bin_v.len() / n_genes;
    let limits = GpuLimits::from_client(client);

    let cube_width = planes_per_gene(batch.max_walked, &limits)? * limits.plane_size_max;

    let anchor: Vec<f64> = bin_v
        .iter()
        .map(|&v| offset_anchor(log_total_sum, v))
        .collect();
    let scalars: Vec<f32> = batch
        .totals
        .iter()
        .map(|&k| k as f32)
        .chain((0..n_genes).map(|g| (guess[g] - anchor[g * n_bins]) as f32))
        .chain(batch.use_table.iter().map(|&t| if t { 1.0 } else { 0.0 }))
        .collect();
    let mut bins: Vec<f32> = bin_v.iter().map(|&v| v as f32).collect();
    bins.extend(
        bin_v
            .iter()
            .zip(&anchor)
            .enumerate()
            .map(|(i, (&v, &a))| ((v * batch.totals[i / n_bins]).ln() - a) as f32),
    );
    bins.extend((0..n_genes * n_bins).map(|i| {
        if i % n_bins == 0 {
            0.0
        } else {
            (anchor[i] - anchor[i - 1]) as f32
        }
    }));

    let log_vs = bins[n_genes * n_bins..2 * n_genes * n_bins].to_vec();
    let scalars = GpuTensor::<R, f32>::from_slice(&scalars, vec![3 * n_genes], client)?;
    let bins = GpuTensor::<R, f32>::from_slice(&bins, vec![3 * n_genes * n_bins], client)?;
    let out = GpuTensor::<R, f32>::empty(vec![6 * n_genes * n_bins], client)?;
    let status = GpuTensor::<R, u32>::empty(vec![n_genes], client)?;

    let (gx, gy) = grid_2d(n_genes as u32, &limits)?;
    let count = checked_cube_count("sweep_grid_gpu", gx, gy, 1, &limits)?;
    unsafe {
        sweep_grid_gpu::launch_unchecked::<f32, R>(
            client,
            count,
            CubeDim::new_1d(cube_width),
            batch.counts.into_tensor_arg(),
            batch.list_offsets.into_tensor_arg(),
            batch.list_cells.into_tensor_arg(),
            batch.list_counts.into_tensor_arg(),
            log_totals.into_tensor_arg(),
            scalars.into_tensor_arg(),
            table_dev.breaks.into_tensor_arg(),
            table_dev.anchors.into_tensor_arg(),
            table_dev.coeffs.into_tensor_arg(),
            table_dev.scalars.into_tensor_arg(),
            table_dev.n_panels,
            bins.into_tensor_arg(),
            out.into_tensor_arg(),
            status.into_tensor_arg(),
            n_genes as u32,
            batch.counts_cells as u32,
            n_bins as u32,
            u32::from(cold_start),
        );
    }

    let status = status.read(client)?;
    if let Some((g, &code)) = status.iter().enumerate().find(|(_, c)| **c != 0) {
        return Err(SanityErrors::GpuOffsetSolveDiverged {
            gene: batch.first + g,
            bin: code as usize - 1,
        });
    }
    let raw = out.clone().read(client)?;
    let host = finish_sweep(&raw, &log_vs, &batch.use_table, table);
    let offsets = anchor
        .iter()
        .zip(&host)
        .map(|(&a, &zeta)| a + zeta)
        .collect();
    Ok(SweepResult {
        device: out,
        bins,
        scalars,
        host,
        offsets,
    })
}

/// Widen the first pass's output to `f64` and add the empty cells' share.
///
/// A table gene's totals come back without it; it is added here from the `f64`
/// table at the `f32` shift the device stopped at, the subtraction it
/// performed itself. The device's `f32` table only steers its solve, and the
/// likelihood's off-root term, `sum_c d_c / v` times order `s`, would carry
/// that table's rounding.
///
/// A host Newton step on the offset from the completed residual was measured
/// on 2026-09-30 and moved no offset by more than `5e-6`, table gene or not,
/// so the device's offsets are used as they are.
///
/// ### Params
///
/// * `raw` - The device output.
/// * `log_vs` - `ln(v s) - a_b` per gene and bin, as the device read it.
/// * `use_table` - Per gene, whether it used the table.
/// * `table` - The run's table, if it has one.
///
/// ### Returns
///
/// The output in `f64`, same layout.
fn finish_sweep(
    raw: &[f32],
    log_vs: &[f32],
    use_table: &[bool],
    table: Option<&ShiftTable>,
) -> Vec<f64> {
    let stride = log_vs.len();
    let n_bins = stride / use_table.len();
    let mut host: Vec<f64> = raw.iter().map(|&x| x as f64).collect();
    let empties: Vec<EmptySums> = (0..stride)
        .into_par_iter()
        .map(|i| match table {
            Some(table) if use_table[i / n_bins] => {
                let shift = (log_vs[i] - raw[i]) as f64;
                table.eval(shift)
            }
            _ => EmptySums::default(),
        })
        .collect();
    for (i, empty) in empties.into_iter().enumerate() {
        host[2 * stride + i] += empty.omega_sq;
        host[4 * stride + i] += empty.log_diag;
        host[5 * stride + i] -= empty.omega;
    }
    host
}

/// Run the second pass over a batch.
///
/// ### Params
///
/// * `batch` - The resident batch.
/// * `log_totals` - `ln T_c` on the device.
/// * `n_cells` - Cells.
/// * `sweep` - The first pass whose bins to integrate over.
/// * `weights` - `W_b` per gene and bin, zero for a dropped bin.
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// `d_c` for every gene and cell, then `e_c`, gene-major.
fn run_marginalise<R: Runtime>(
    batch: &Batch<R>,
    log_totals: &GpuTensor<R, f32>,
    n_cells: usize,
    sweep: &SweepResult<R>,
    weights: &[f32],
    client: &ComputeClient<R>,
) -> Result<Vec<f32>, SanityErrors> {
    let n_genes = batch.n_genes;
    let n_bins = weights.len() / n_genes;
    let limits = GpuLimits::from_client(client);

    let weights = GpuTensor::<R, f32>::from_slice(weights, vec![n_genes * n_bins], client)?;
    let out = GpuTensor::<R, f32>::empty(vec![2 * n_genes * n_cells], client)?;

    let width = resolve_workgroup_size(MARGINALISE_WORKGROUP, &limits);
    let blocks_per_gene = (n_cells as u32).div_ceil(width);
    let (gx, gy) = grid_2d(blocks_per_gene * n_genes as u32, &limits)?;
    let count = checked_cube_count("marginalise_gpu", gx, gy, 1, &limits)?;
    unsafe {
        marginalise_gpu::launch_unchecked::<f32, R>(
            client,
            count,
            CubeDim::new_1d(width),
            batch.counts.into_tensor_arg(),
            log_totals.into_tensor_arg(),
            sweep.scalars.into_tensor_arg(),
            sweep.bins.into_tensor_arg(),
            sweep.device.into_tensor_arg(),
            weights.into_tensor_arg(),
            out.into_tensor_arg(),
            n_genes as u32,
            n_cells as u32,
            n_bins as u32,
            blocks_per_gene,
        );
    }
    Ok(out.read(client)?)
}

/// Log marginal likelihood of one bin from the first pass's sums, up to a
/// constant shared by every bin.
///
/// SI eq. 20 and 33, in the rewritten form of the module doc. The dropped
/// constant is `sum_c k_c ln(k_c T_c / s) - 0.5 sum_{k_c > 0} ln k_c`, which
/// the softmax ignores.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `s` - `K`.
/// * `n_cells` - Cells.
/// * `n_expressed` - Cells with a non-zero count.
/// * `curvature` - `S_A`.
/// * `sum_sq` - `sum_c d_c^2`.
/// * `sum_kw` - `sum_{k > 0} k ln(omega / (v k))`.
/// * `sum_log1p` - `sum_{k = 0} ln(1 + omega) + sum_{k > 0} ln((1 + omega) / (v k))`.
/// * `sum_d` - The offset residual the solve stopped at, `sum_c d_c`.
///
/// ### Returns
///
/// `ln P(k | v)` minus the shared constant.
#[allow(clippy::too_many_arguments)]
fn log_marginal(
    v: f64,
    s: f64,
    n_cells: f64,
    n_expressed: f64,
    curvature: f64,
    sum_sq: f64,
    sum_kw: f64,
    sum_log1p: f64,
    sum_d: f64,
) -> f64 {
    let log_v = v.ln();
    // More details here:
    // SI eq. 20 has `-s ln(sum_c T_c e^{d_c})`, which is `-s z` only at an
    // exact root of the offset residual. `f32` stops a few ulp of `z` short,
    // leaving `sum_c d_c` of order `1e-5 S_A`, and the gap is first order in
    // it: `sum_c T_c e^{d_c} = e^z (1 - sum_c d_c / (v s))`. Keeping the term
    // makes the likelihood the objective at the point actually reached, whose
    // error is second order.
    let off_root = -s * (-sum_d / (v * s)).ln_1p();
    let log_star = -0.5 * n_cells * log_v - 0.5 * sum_sq / v + sum_kw + off_root;
    let log_det = curvature.ln() - (v * s).ln() + sum_log1p + n_expressed * log_v - n_cells * log_v;
    log_star - 0.5 * log_det
}

/// Every bin's log marginal likelihood and offset, from the first pass.
///
/// ### Params
///
/// * `batch` - The batch the pass ran over.
/// * `sweep` - The first pass, over the whole grid.
/// * `grid` - The variance grid.
/// * `n_cells` - Cells.
///
/// ### Returns
///
/// Per gene, `ln P(k | v_b)` up to a per-gene constant, and `z(v_b)`.
fn bin_likelihoods<R: Runtime>(
    batch: &Batch<R>,
    sweep: &SweepResult<R>,
    grid: &[f64],
    n_cells: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n_bins = grid.len();
    let stride = batch.n_genes * n_bins;
    let h = &sweep.host;
    (0..batch.n_genes)
        .into_par_iter()
        .map(|g| {
            let mut log_lik = vec![0.0; n_bins];
            let mut offsets = vec![0.0; n_bins];
            for (b, &v) in grid.iter().enumerate() {
                let i = g * n_bins + b;
                offsets[b] = sweep.offsets[i];
                log_lik[b] = log_marginal(
                    v,
                    batch.totals[g],
                    n_cells as f64,
                    batch.n_expressed[g] as f64,
                    h[stride + i],
                    h[2 * stride + i],
                    h[3 * stride + i],
                    h[4 * stride + i],
                    h[5 * stride + i],
                );
            }
            (log_lik, offsets)
        })
        .collect()
}

/// The most probable bin, with near ties settled in `f64` on the CPU.
///
/// The argmax is discrete, so an `f32` error in the log likelihood far smaller
/// than anything [`VarianceRule::Marginalise`] notices can still move it to
/// another bin when the posterior on `v` is flat. Every bin within
/// [`tie_margin`] of the device's best is re-solved exactly;
/// each costs one offset solve and one Laplace fit, a small fraction of the
/// gene's full grid.
///
/// ### Params
///
/// * `counts` - The count matrix.
/// * `gene` - Input index of the gene.
/// * `log_totals` - `ln T_c` for every cell.
/// * `grid` - The variance grid.
/// * `s` - `K`.
/// * `log_lik` - The device's log likelihood per bin.
/// * `offsets` - The device's offset per bin, used as warm starts.
/// * `table` - The run's empty-cell table; a gene at most
///   [`SPARSE_MAX_DENSITY`] dense re-solves sparse against it, as on the CPU.
///
/// ### Returns
///
/// The winning bin and its `f64` offset, or a solver failure.
#[allow(clippy::too_many_arguments)]
fn resolve_max_posterior(
    counts: &CountMatrix,
    gene: usize,
    log_totals: &[f64],
    grid: &[f64],
    s: f64,
    log_lik: &[f64],
    offsets: &[f64],
    table: Option<&ShiftTable>,
) -> Result<(usize, f64), SanityErrors> {
    let peak = log_lik.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let candidates: Vec<usize> = (0..grid.len())
        .filter(|&b| log_lik[b] >= peak - tie_margin(log_totals.len()))
        .collect();
    if let [only] = candidates[..] {
        return Ok((only, offsets[only]));
    }

    let n_cells = log_totals.len();
    let (indices, values) = counts.gene(gene);
    let mut best = (candidates[0], f64::NEG_INFINITY, offsets[candidates[0]]);
    if let Some(table) =
        table.filter(|_| indices.len() as f64 <= SPARSE_MAX_DENSITY * n_cells as f64)
    {
        let mut cells = NonzeroCells::new(indices.len());
        cells.load(indices, values, log_totals);
        for b in candidates {
            let (l, z) =
                log_marginal_at_sparse(grid[b], s, offsets[b], table, log_totals, &mut cells)?;
            if l > best.1 {
                best = (b, l, z);
            }
        }
        return Ok((best.0, best.2));
    }

    let mut dense = vec![0.0; n_cells];
    for (&i, &k) in indices.iter().zip(values) {
        dense[i as usize] = k as f64;
    }
    let mut omega = vec![0.0; n_cells];
    let mut log_omega = vec![0.0; n_cells];
    let mut state = None;
    // `candidates` ascends in `v`, so each solve warm starts from the last.
    for b in candidates {
        let (l, z) = log_marginal_at(
            grid[b],
            s,
            &dense,
            log_totals,
            offsets[b],
            &mut state,
            &mut omega,
            &mut log_omega,
        )?;
        if l > best.1 {
            best = (b, l, z);
        }
    }
    Ok((best.0, best.2))
}

//////////////
// Frontend //
//////////////

/// Run Sanity on the GPU.
///
/// Same inference as [`crate::sanity`], evaluated in `f32` on the device with
/// the likelihood assembled in `f64` on the host. See the module doc for the
/// precision this costs. Genes are processed in batches bounded by
/// `GPU_BATCH_BYTES` and the device's per-binding limit.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes.
/// * `params` - Run parameters, or [`SanityParams::default`].
/// * `client` - CubeCL compute client.
///
/// ### Returns
///
/// The same output as [`crate::sanity`]. Values are `f32` precision whatever `T`.
pub fn sanity_gpu<T: SanityFloat, R: Runtime>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
    client: &ComputeClient<R>,
) -> Result<SanityOutput<T>, SanityErrors> {
    sanity_gpu_batched(counts, cell_totals, params, client, GPU_BATCH_BYTES, |_| {
        true
    })
}

/// Run Sanity on the GPU and store only the genes a predicate keeps.
///
/// The GPU twin of [`crate::sanity_select`]. Each batch comes back to the host
/// anyway, so the predicate runs there, per gene, before anything is stored:
/// resident output is one batch plus the kept genes, not `2 * n_genes *
/// n_cells` values.
///
/// The [`GeneView`] rows are the device's `f32` values widened to `f64`, so a
/// predicate sitting on a threshold can decide a borderline gene differently
/// from the CPU run.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes.
/// * `params` - Run parameters, or [`SanityParams::default`].
/// * `client` - CubeCL compute client.
/// * `keep` - Returns `true` for a gene to store. Called once per gene, from
///   any Rayon worker.
///
/// ### Returns
///
/// The kept genes in input order, with their input indices in
/// [`SanityOutput::genes`]. Keeping nothing is not an error; the output is then
/// empty.
pub fn sanity_gpu_select<T, R, F>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
    client: &ComputeClient<R>,
    keep: F,
) -> Result<SanityOutput<T>, SanityErrors>
where
    T: SanityFloat,
    R: Runtime,
    F: Fn(GeneView<'_>) -> bool + Sync,
{
    let out = sanity_gpu_batched(counts, cell_totals, params, client, GPU_BATCH_BYTES, keep)?;
    if params.unwrap_or_default().verbosity.normal_verbosity() {
        println!("  Kept {} / {} genes", out.n_genes, counts.n_genes());
    }
    Ok(out)
}

/// [`sanity_gpu`] with the batch budget as a parameter.
///
/// ### Params
///
/// * `counts` - Raw UMI counts, gene-major sparse.
/// * `cell_totals` - Total UMI count of every cell over *all* genes.
/// * `params` - Run parameters, or [`SanityParams::default`].
/// * `client` - CubeCL compute client.
/// * `batch_bytes` - Ceiling on the largest buffer of one batch.
/// * `keep` - Per-gene predicate, as for [`sanity_gpu_select`].
///
/// ### Returns
///
/// As [`sanity_gpu_select`].
fn sanity_gpu_batched<T, R, F>(
    counts: &CountMatrix,
    cell_totals: &[f64],
    params: Option<SanityParams>,
    client: &ComputeClient<R>,
    batch_bytes: u64,
    keep: F,
) -> Result<SanityOutput<T>, SanityErrors>
where
    T: SanityFloat,
    R: Runtime,
    F: Fn(GeneView<'_>) -> bool + Sync,
{
    let params = params.unwrap_or_default();
    let n_cells = counts.n_cells();
    let n_genes = counts.n_genes();
    let (grid, log_totals, log_total_sum) = prepare_run(counts, cell_totals, &params)?;
    let limits = GpuLimits::from_client(client);

    let row_bytes = 2 * n_cells as u64 * size_of::<f32>() as u64;
    let budget = batch_bytes.min(limits.max_binding_bytes);
    let batch_genes = ((budget / row_bytes) as usize).clamp(1, n_genes.max(1));

    let verbose = params.verbosity.normal_verbosity();
    let detailed = params.verbosity.detailed_verbosity();
    let n_batches = n_genes.div_ceil(batch_genes);
    if verbose {
        print_run_header("GPU", n_genes, n_cells, &params);
        println!("  Batches: {n_batches}, up to {batch_genes} genes each");
    }
    let start = Instant::now();

    let table = shift_table(counts, &log_totals, &params);
    let table_dev = DeviceTable::<R>::new(table.as_ref(), client)?;
    let log_totals_f32: Vec<f32> = log_totals.iter().map(|&x| x as f32).collect();
    let log_totals_dev = GpuTensor::<R, f32>::from_slice(&log_totals_f32, vec![n_cells], client)?;

    let mut out = SanityOutput {
        log_fold_changes: Vec::new(),
        error_bars: Vec::new(),
        mean_log_quotient: Vec::new(),
        mean_log_quotient_error: Vec::new(),
        variance: Vec::new(),
        genes: Vec::new(),
        n_genes: 0,
        n_cells,
    };

    let mut first = 0;
    while first < n_genes {
        // Stage wall times for `Detailed`. `run_sweep` and `run_marginalise`
        // read back to the host, so each lap ends synchronised; an upload the
        // runtime defers lands in the sweep after it rather than in `stage`.
        let mut laps: Vec<(&str, Duration)> = Vec::new();
        let mut t = Instant::now();
        let mut lap = |name| {
            laps.push((name, t.elapsed()));
            t = Instant::now();
        };

        let n = batch_genes.min(n_genes - first);
        let batch = stage_batch(counts, first, n, table.is_some(), client)?;
        let ks = &batch.totals;
        lap("stage");

        let (sweep, weights, summaries) = match params.variance_rule {
            VarianceRule::Fixed(v) => {
                let guess = vec![log_total_sum + 0.5 * v; n];
                let sweep = run_sweep(
                    &batch,
                    &log_totals_dev,
                    table.as_ref(),
                    &table_dev,
                    log_total_sum,
                    &vec![v; n],
                    &guess,
                    true,
                    client,
                )?;
                lap("sweep");
                let summaries: Vec<(f64, f64, f64)> = (0..n)
                    .map(|g| (digamma(ks[g]) - sweep.offsets[g], trigamma(ks[g]).sqrt(), v))
                    .collect();
                (sweep, vec![1.0f32; n], summaries)
            }
            rule => {
                let bin_v: Vec<f64> = (0..n).flat_map(|_| grid.values.iter().copied()).collect();
                let guess = vec![log_total_sum + 0.5 * grid.values[0]; n];
                let sweep = run_sweep(
                    &batch,
                    &log_totals_dev,
                    table.as_ref(),
                    &table_dev,
                    log_total_sum,
                    &bin_v,
                    &guess,
                    true,
                    client,
                )?;
                lap("sweep");

                let per_gene: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> =
                    bin_likelihoods(&batch, &sweep, &grid.values, n_cells)
                        .into_par_iter()
                        .map(|(log_lik, offsets)| {
                            let mut weights = vec![0.0; log_lik.len()];
                            posterior_weights(&log_lik, &mut weights)?;
                            Ok((weights, offsets, log_lik))
                        })
                        .collect::<Result<_, SanityErrors>>()?;

                if matches!(rule, VarianceRule::Marginalise) {
                    let summaries = per_gene
                        .iter()
                        .enumerate()
                        .map(|(g, (weights, offsets, _))| {
                            let summary = marginal_summary(ks[g], &grid.values, weights, offsets).0;
                            (
                                summary.mean_log_quotient,
                                summary.mean_log_quotient_error,
                                summary.variance,
                            )
                        })
                        .collect();
                    let weights: Vec<f32> = per_gene
                        .iter()
                        .flat_map(|(w, _, _)| w.iter())
                        .map(|&w| {
                            if w < MARGINALISE_MIN_WEIGHT {
                                0.0
                            } else {
                                w as f32
                            }
                        })
                        .collect();
                    lap("host");
                    (sweep, weights, summaries)
                } else {
                    let targets: Vec<(f64, f64, f64)> = per_gene
                        .par_iter()
                        .enumerate()
                        .map(|(g, (w, z, log_lik))| {
                            let target = collapse_target(rule, &grid.values, w, z);
                            if !matches!(rule, VarianceRule::MaxPosterior) {
                                return Ok(target);
                            }
                            let (b, z) = resolve_max_posterior(
                                counts,
                                first + g,
                                &log_totals,
                                &grid.values,
                                ks[g],
                                log_lik,
                                z,
                                table.as_ref(),
                            )?;
                            Ok((grid.values[b], z, target.2))
                        })
                        .collect::<Result<_, SanityErrors>>()?;
                    let bin_v: Vec<f64> = targets.iter().map(|t| t.0).collect();
                    let guess: Vec<f64> = targets.iter().map(|t| t.1).collect();
                    lap("host");
                    drop(sweep);
                    let point = run_sweep(
                        &batch,
                        &log_totals_dev,
                        table.as_ref(),
                        &table_dev,
                        log_total_sum,
                        &bin_v,
                        &guess,
                        false,
                        client,
                    )?;
                    lap("re-solve");
                    let summaries = (0..n)
                        .map(|g| {
                            (
                                digamma(ks[g]) - point.offsets[g],
                                trigamma(ks[g]).sqrt(),
                                targets[g].2,
                            )
                        })
                        .collect();
                    (point, vec![1.0f32; n], summaries)
                }
            }
        };

        let rows = run_marginalise(&batch, &log_totals_dev, n_cells, &sweep, &weights, client)?;
        let (lfc, eb) = rows.split_at(n * n_cells);
        let kept: Vec<bool> = (0..n)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_cells], vec![0.0f64; n_cells]),
                |(d, e), g| {
                    let lo = g * n_cells;
                    for c in 0..n_cells {
                        d[c] = lfc[lo + c] as f64;
                        e[c] = eb[lo + c] as f64;
                    }
                    let (m, dm, v) = summaries[g];
                    keep(GeneView {
                        log_fold_changes: d,
                        error_bars: e,
                        mean_log_quotient: m,
                        mean_log_quotient_error: dm,
                        variance: v,
                    })
                },
            )
            .collect();
        for (g, &(m, dm, v)) in summaries.iter().enumerate() {
            if !kept[g] {
                continue;
            }
            let lo = g * n_cells;
            out.log_fold_changes
                .extend(lfc[lo..lo + n_cells].iter().map(|&x| narrow::<T>(x as f64)));
            out.error_bars
                .extend(eb[lo..lo + n_cells].iter().map(|&x| narrow::<T>(x as f64)));
            out.mean_log_quotient.push(narrow(m));
            out.mean_log_quotient_error.push(narrow(dm));
            out.variance.push(narrow(v));
            out.genes.push(first + g);
        }
        lap("second pass");

        if detailed {
            let split: Vec<String> = laps
                .iter()
                .map(|(name, d)| format!("{name} {d:.2?}"))
                .collect();
            println!(
                "  Batch {}/{n_batches} ({n} genes): {}",
                first / batch_genes + 1,
                split.join(", ")
            );
        }
        if verbose {
            report_decile_progress(first + n, first, n_genes, "genes", start.elapsed());
        }
        first += n;
    }

    out.n_genes = out.genes.len();
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use crate::simulate::{SimulationParams, simulate};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn test_gpu_select_keeps_the_full_run_rows_it_selects() {
        let sim = simulate(Some(SimulationParams {
            n_genes: 50,
            n_cells: 2000,
            library_size: 500.0,
            seed: 5,
            ..Default::default()
        }))
        .expect("simulates");
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let full: SanityOutput<f32> =
            sanity_gpu(&sim.counts, &sim.cell_totals, None, &client).expect("runs");

        let mut sorted = full.variance.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        // between two stored values, so the f64 the predicate sees and the f32
        // stored here cannot fall on opposite sides of it
        let mid = sorted.len() / 2;
        let cut = 0.5 * (sorted[mid - 1] as f64 + sorted[mid] as f64);
        let n_cells = full.n_cells;
        // seven genes per batch, so the selection crosses batch boundaries
        let row_bytes = 2 * n_cells as u64 * size_of::<f32>() as u64;
        let picked: SanityOutput<f32> = sanity_gpu_batched(
            &sim.counts,
            &sim.cell_totals,
            None,
            &client,
            7 * row_bytes,
            |g| g.variance > cut,
        )
        .expect("runs");

        let expected: Vec<usize> = (0..full.n_genes)
            .filter(|&g| full.variance[g] as f64 > cut)
            .collect();
        assert_eq!(picked.genes, expected);
        assert_eq!(picked.n_genes, expected.len());
        for (row, &g) in expected.iter().enumerate() {
            let (a, b) = (row * n_cells, g * n_cells);
            assert_eq!(
                picked.log_fold_changes[a..a + n_cells],
                full.log_fold_changes[b..b + n_cells]
            );
            assert_eq!(
                picked.error_bars[a..a + n_cells],
                full.error_bars[b..b + n_cells]
            );
            assert_eq!(picked.variance[row], full.variance[g]);
        }

        let none: SanityOutput<f32> =
            sanity_gpu_select(&sim.counts, &sim.cell_totals, None, &client, |_| false)
                .expect("runs");
        assert_eq!(none.n_genes, 0);
        assert!(none.log_fold_changes.is_empty());
    }

    #[test]
    fn test_gpu_batching_does_not_change_the_output() {
        let sim = simulate(Some(SimulationParams {
            n_genes: 50,
            n_cells: 2000,
            library_size: 500.0,
            seed: 3,
            ..Default::default()
        }))
        .expect("simulates");
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let whole: SanityOutput<f32> =
            sanity_gpu(&sim.counts, &sim.cell_totals, None, &client).expect("runs in one batch");
        // Seven genes per batch, so the last batch is ragged.
        let row_bytes = 2 * 2000 * size_of::<f32>() as u64;
        let split: SanityOutput<f32> = sanity_gpu_batched(
            &sim.counts,
            &sim.cell_totals,
            None,
            &client,
            7 * row_bytes,
            |_| true,
        )
        .expect("runs in batches");
        assert_eq!(split.log_fold_changes, whole.log_fold_changes);
        assert_eq!(split.error_bars, whole.error_bars);
        assert_eq!(split.mean_log_quotient, whole.mean_log_quotient);
        assert_eq!(split.variance, whole.variance);
    }

    /// Worst device error in a log likelihood gap, over simulated genes.
    ///
    /// The error that matters for the argmax: how far the device moves one
    /// bin's log likelihood relative to the CPU's best bin, over the bins close
    /// enough to the peak to compete for it. A library size of 500 over 100
    /// genes puts nearly every gene past the density gate, so only 25 reaches
    /// the table path.
    fn worst_gap_error(n_genes: usize, n_cells: usize, library_size: f64) -> f64 {
        let sim = simulate(Some(SimulationParams {
            n_genes,
            n_cells,
            library_size,
            seed: 5,
            ..Default::default()
        }))
        .expect("simulates");
        let params = SanityParams::default();
        let (grid, log_totals, log_total_sum) =
            prepare_run(&sim.counts, &sim.cell_totals, &params).expect("valid");
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let n = sim.counts.n_genes();
        let log_totals_f32: Vec<f32> = log_totals.iter().map(|&x| x as f32).collect();
        let log_totals_dev =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&log_totals_f32, vec![n_cells], &client)
                .expect("uploads");
        let table = shift_table(&sim.counts, &log_totals, &params);
        let table_dev = DeviceTable::new(table.as_ref(), &client).expect("uploads");
        let batch = stage_batch(&sim.counts, 0, n, table.is_some(), &client).expect("stages");
        let bin_v: Vec<f64> = (0..n).flat_map(|_| grid.values.iter().copied()).collect();
        let guess = vec![log_total_sum + 0.5 * grid.values[0]; n];
        let sweep = run_sweep(
            &batch,
            &log_totals_dev,
            table.as_ref(),
            &table_dev,
            log_total_sum,
            &bin_v,
            &guess,
            true,
            &client,
        )
        .expect("sweeps");
        let device = bin_likelihoods(&batch, &sweep, &grid.values, n_cells);

        let worst = device
            .par_iter()
            .enumerate()
            .map(|(g, (gpu, offsets))| {
                let mut dense = vec![0.0; n_cells];
                let (indices, values) = sim.counts.gene(g);
                for (&i, &k) in indices.iter().zip(values) {
                    dense[i as usize] = k as f64;
                }
                let mut omega = vec![0.0; n_cells];
                let mut log_omega = vec![0.0; n_cells];
                let mut state = None;
                let cpu: Vec<f64> = grid
                    .values
                    .iter()
                    .zip(offsets)
                    .map(|(&v, &z)| {
                        log_marginal_at(
                            v,
                            batch.totals[g],
                            &dense,
                            &log_totals,
                            z,
                            &mut state,
                            &mut omega,
                            &mut log_omega,
                        )
                        .expect("solves")
                        .0
                    })
                    .collect();
                let best = (0..cpu.len())
                    .max_by(|&a, &b| cpu[a].partial_cmp(&cpu[b]).expect("finite"))
                    .expect("non-empty");
                (0..cpu.len())
                    .filter(|&b| cpu[b] >= cpu[best] - 10.0)
                    .map(|b| ((gpu[b] - gpu[best]) - (cpu[b] - cpu[best])).abs())
                    .fold(0.0f64, f64::max)
            })
            .reduce(|| 0.0, f64::max);
        println!(
            "{n_genes} genes x {n_cells} cells, library {library_size}: worst device error in a log likelihood gap {worst:e}"
        );
        worst
    }

    #[test]
    fn test_gpu_bin_likelihood_error_is_inside_the_tie_margin() {
        let margin = tie_margin(20_000);
        for library_size in [500.0, 25.0] {
            let worst = worst_gap_error(100, 20_000, library_size);
            assert!(
                worst <= 0.5 * margin,
                "library {library_size}: device error {worst:e} is not safely inside the tie margin {margin:e}"
            );
        }
    }

    #[test]
    #[ignore = "a 200k cell CPU reference; run with --ignored"]
    fn test_gpu_bin_likelihood_error_is_inside_the_tie_margin_at_scale() {
        let margin = tie_margin(200_000);
        for library_size in [500.0, 25.0] {
            let worst = worst_gap_error(50, 200_000, library_size);
            assert!(
                worst <= 0.5 * margin,
                "library {library_size}: device error {worst:e} is not safely inside the tie margin {margin:e}"
            );
        }
    }
}
