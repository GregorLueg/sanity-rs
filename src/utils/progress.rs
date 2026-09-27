//! Progress lines for long runs.

use std::time::Duration;

////////////
// Consts //
////////////

/// Step size, in percent, between progress reports.
///
/// Reporting on decile crossings rather than per gene bounds a run at ten
/// lines, whether it walks ten genes or thirty thousand.
const PROGRESS_STEP_PCT: usize = 10;

//////////////
// Progress //
//////////////

/// Prints a progress line whenever a sweep crosses a decile of its work.
///
/// Cheap enough to call per gene: it formats nothing unless `done` and
/// `prev_done` fall either side of a [`PROGRESS_STEP_PCT`] boundary, or the
/// sweep has just finished. The verbosity check stays with the caller.
///
/// ### Params
///
/// * `done` - Units of work finished, including the one just completed.
/// * `prev_done` - Units of work finished before it.
/// * `total` - Units of work in the whole sweep. Nothing prints if this is `0`.
/// * `unit` - What is being counted, e.g. `"genes"`. Printed as given.
/// * `elapsed` - Time since the sweep started.
pub(crate) fn report_decile_progress(
    done: usize,
    prev_done: usize,
    total: usize,
    unit: &str,
    elapsed: Duration,
) {
    if total == 0 {
        return;
    }

    let pct = done * 100 / total;
    let prev_pct = prev_done * 100 / total;

    if pct / PROGRESS_STEP_PCT > prev_pct / PROGRESS_STEP_PCT || done == total {
        println!("  Progress: {pct}% ({done} / {total} {unit}, {elapsed:.2?})");
    }
}
