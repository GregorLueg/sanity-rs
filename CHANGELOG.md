# Changelog

## v0.2.1

### Features

* The GPU-accelerated form of Sanity also got a filter gene option to only
  keep these genes.

## v0.2.0

### Features

* Progress reporting via `SanityParams::verbosity` and the new `Verbosity`
  enum (`Quiet`, `Normal`, `Detailed`), on `sanity`, `sanity_select` and
  `sanity_gpu`. `Normal` prints a header and a line at every tenth of the
  genes; `Detailed` adds a per-batch stage split on the GPU. Default is
  `Quiet`.
* Breaking: `SanityParams::new` takes a trailing `verbosity`. Struct literals
  that end in `..SanityParams::default()` compile unchanged.

### Docs

* README: the hand-over to Bonsai goes through bonsai-rs's
  `ingest::from_sanity_output`, which reads `log_fold_changes`, not the log
  transcription quotients.

## v0.1.0

### Features

* GPU-accelerated Sanity added via the wgpu/cubecl framework, behind the `gpu`
  feature. `sanity_gpu` takes and returns what `sanity` does and supports every
  variance rule. The device works in `f32`; the maths is rearranged so no sum
  cancels, the offset is solved against an `f64` anchor and the likelihood over
  the variance grid is assembled in `f64` on the host.
* `MaxPosterior` on the GPU settles near ties between bins in `f64` on the CPU,
  so it lands on the same bin as the CPU path.
* `gpu-tests` feature with CPU parity tests for every rule, and a GPU lane in
  CI.

## v0.0.1

### Features

* Clean-room Rust implementation of Sanity from `docs/SPEC.md`: `sanity`,
  `sanity_select`, the four variance rules and the simulator.
