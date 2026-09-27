//! Run parameters and the variance grid.

////////////
// Consts //
////////////

/// Lower bound of the variance grid.
///
/// The smallest variance that is distinguishable from zero given Poisson
/// sampling. A gene observed at `m` UMIs per cell carries log-scale sampling
/// noise of standard deviation about `1 / sqrt(m)`; the most favourable genes in
/// a droplet experiment sit near `m = 1000`, giving `0.032`, so variances below
/// `1e-3` are hidden under the noise floor of even the best-measured gene.
pub const DEFAULT_VARIANCE_MIN: f64 = 1e-3;

/// Upper bound of the variance grid.
///
/// `sqrt(v)` is the standard deviation of the log transcription quotient, so
/// `v = 50` spans a factor of `exp(2 * 7.07) = 1.2e6` between the sixteenth and
/// eighty-fourth percentiles. With library sizes of order `1e4` the low end of
/// that range is below one expected UMI in every cell, so no larger variance is
/// resolvable.
pub const DEFAULT_VARIANCE_MAX: f64 = 50.0;

/// Number of bins, equally spaced in `ln v`.
///
/// Measured 2026-09-13, under the earlier `s = K + 1` prior (SPEC section 1);
/// not yet re-measured under `s = K`. Refining the grid along a `2^k + 1` ladder, so that each
/// finer grid contains every point of the coarser ones, 161 is the coarsest rung
/// at which a further doubling moves every estimate by less than 1% of the error
/// bar the method itself reports for that estimate. Checked on 500 simulated
/// genes over 500 cells and on 300 real genes over 498 cells; 81 bins suffices
/// for the real data but not for the simulated, which carries wider variances.
///
/// This applies to [`VarianceRule::Marginalise`]. [`VarianceRule::MaxPosterior`]
/// does not converge under grid refinement at all, by construction: it selects a
/// bin by discrete argmax, so a finer grid keeps changing which bin wins. At 641
/// bins it was still moving by 3% of an error bar.
pub const DEFAULT_VARIANCE_BINS: usize = 161;

/// How the per-gene variance enters the final estimates.
///
/// SPEC section 7. Only [`VarianceRule::Marginalise`] integrates over the
/// posterior on `v`; the rest collapse it to a single value first, which costs
/// accuracy in the error bars and saves a second sweep of the grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VarianceRule {
    /// Integrate over the full posterior on `v`, SI eq. 39 and 42. The default.
    Marginalise,
    /// Collapse to the posterior mean `sum_b W_b v_b` and re-solve there.
    PosteriorMean,
    /// Collapse to the most probable bin, `argmax_b L_b`.
    ///
    /// The only rule whose output depends on the grid no matter how fine it is:
    /// the argmax is discrete, so refining the grid keeps moving the selected
    /// bin. Use it for speed, not for a number you intend to quote.
    ///
    /// `d_c` and `e_c` come from the selected bin, but the reported `variance`
    /// stays the posterior mean over the whole grid, by design: the argmax is
    /// the wrong summary of a posterior to quote back to a caller.
    MaxPosterior,
    /// Use the supplied variance for every gene; no grid and no scan.
    Fixed(f64),
}

impl VarianceRule {
    /// Whether this rule needs the weights over the whole grid.
    ///
    /// ### Returns
    ///
    /// `false` only for [`VarianceRule::Fixed`], which skips the scan entirely.
    #[inline]
    pub(crate) fn needs_grid(&self) -> bool {
        !matches!(self, VarianceRule::Fixed(_))
    }
}

/// How much a run prints to stdout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Verbosity {
    /// Nothing at all. The default.
    #[default]
    Quiet,
    /// A header, progress over genes at every tenth of the run, and the total
    /// time.
    Normal,
    /// As [`Verbosity::Normal`], plus a per-batch stage split on the GPU. The
    /// CPU path prints nothing extra.
    Detailed,
}

impl Verbosity {
    /// Whether normal or detailed output is on.
    ///
    /// ### Returns
    ///
    /// `true` for [`Verbosity::Normal`] and [`Verbosity::Detailed`].
    #[inline]
    pub fn normal_verbosity(&self) -> bool {
        matches!(self, Verbosity::Normal | Verbosity::Detailed)
    }

    /// Whether detailed output is on.
    ///
    /// ### Returns
    ///
    /// `true` for [`Verbosity::Detailed`] only.
    #[inline]
    pub fn detailed_verbosity(&self) -> bool {
        matches!(self, Verbosity::Detailed)
    }
}

/// Parameters for a Sanity run.
#[derive(Clone, Copy, Debug)]
pub struct SanityParams {
    /// How the per-gene variance enters the final estimates.
    pub variance_rule: VarianceRule,
    /// Lower bound of the variance grid.
    pub variance_min: f64,
    /// Upper bound of the variance grid.
    pub variance_max: f64,
    /// Number of grid bins, equally spaced in `ln v`.
    pub variance_bins: usize,
    /// How much the run prints.
    pub verbosity: Verbosity,
}

impl SanityParams {
    /// Construct a parameter set.
    ///
    /// ### Params
    ///
    /// * `variance_rule` - How the per-gene variance enters the estimates.
    /// * `variance_min` - Lower bound of the variance grid.
    /// * `variance_max` - Upper bound of the variance grid.
    /// * `variance_bins` - Number of grid bins.
    /// * `verbosity` - How much the run prints.
    ///
    /// ### Returns
    ///
    /// The parameter set. Validation happens at the entry point, not here.
    pub fn new(
        variance_rule: VarianceRule,
        variance_min: f64,
        variance_max: f64,
        variance_bins: usize,
        verbosity: Verbosity,
    ) -> Self {
        Self {
            variance_rule,
            variance_min,
            variance_max,
            variance_bins,
            verbosity,
        }
    }
}

impl Default for SanityParams {
    fn default() -> Self {
        Self {
            variance_rule: VarianceRule::Marginalise,
            variance_min: DEFAULT_VARIANCE_MIN,
            variance_max: DEFAULT_VARIANCE_MAX,
            variance_bins: DEFAULT_VARIANCE_BINS,
            verbosity: Verbosity::Normal,
        }
    }
}

/// The variance grid, equally spaced in `ln v`.
///
/// The scale prior on `v` is uniform in `ln v` (SI eq. 34 and the paragraph
/// above it), so equal spacing in the logarithm makes a flat weighting over bins
/// *be* the prior. Nothing else in the crate reinstates it.
#[derive(Clone, Debug)]
pub struct VarianceGrid {
    /// Bin centres, ascending.
    pub values: Vec<f64>,
}

impl VarianceGrid {
    /// Build a grid from bounds and a bin count.
    ///
    /// ### Params
    ///
    /// * `min` - Lower bound, strictly positive.
    /// * `max` - Upper bound, greater than `min`.
    /// * `bins` - Number of bins. A single bin sits at the geometric mean.
    ///
    /// ### Returns
    ///
    /// The grid. Bounds are assumed validated by the caller.
    pub fn new(min: f64, max: f64, bins: usize) -> Self {
        if bins <= 1 {
            return Self {
                values: vec![(min * max).sqrt()],
            };
        }
        let log_min = min.ln();
        let step = (max.ln() - log_min) / (bins - 1) as f64;
        Self {
            values: (0..bins)
                .map(|b| (log_min + step * b as f64).exp())
                .collect(),
        }
    }

    /// Number of bins.
    ///
    /// ### Returns
    ///
    /// The bin count.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the grid is empty.
    ///
    /// ### Returns
    ///
    /// `true` if there are no bins, which the constructor never produces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_grid_spans_the_bounds() {
        let grid = VarianceGrid::new(1e-3, 50.0, 160);
        assert_eq!(grid.len(), 160);
        assert_relative_eq!(grid.values[0], 1e-3, max_relative = 1e-12);
        assert_relative_eq!(grid.values[159], 50.0, max_relative = 1e-12);
    }

    #[test]
    fn test_grid_is_uniform_in_log() {
        let grid = VarianceGrid::new(1e-3, 50.0, 160);
        let ratio = grid.values[1] / grid.values[0];
        for b in 1..grid.len() {
            assert_relative_eq!(
                grid.values[b] / grid.values[b - 1],
                ratio,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_single_bin_grid_is_the_geometric_mean() {
        let grid = VarianceGrid::new(1e-3, 50.0, 1);
        assert_eq!(grid.len(), 1);
        assert_relative_eq!(
            grid.values[0],
            (1e-3f64 * 50.0).sqrt(),
            max_relative = 1e-12
        );
    }
}
