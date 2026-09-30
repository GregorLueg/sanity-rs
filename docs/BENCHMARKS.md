# Benchmarks

Numbers for the current version, v0.3.0. Older ones live in the git history of
this file.

Machine: Apple M1 Max, 10 cores, 64 GB, macOS 26.6.2, release profile. Input is
this crate's simulator throughout. macOS background daemons kept the load
average between 7 and 80 during the 2026-09-30 runs, so treat single timings as
good to 10 to 20%.

## Against the reference binary

Black-box comparison with the reference Sanity binary (v2.0.0, built from
`jmbreda/Sanity`), under the terms in `docs/PROVENANCE.md`. The harness lives
outside this repository at `~/repos/others/sanity-comparison`. Both read the
same Matrix Market file and write text output. Run: `Marginalise` /
`-v_m MARG`, 160 bins over `[1e-3, 50]`. Wall time and peak RSS are for the
whole process, from `/usr/bin/time -l`.

The reference timings are from 2026-09-24, same machine; the binary has not
changed since. Ours are from 2026-09-30.

490 genes by 2000 cells (500 requested, 10 came back empty), library size 125,
77 359 stored counts (7.9% dense):

| threads | wall, ours | wall, reference | speedup | peak RSS, ours | peak RSS, reference |
|---|---|---|---|---|---|
| 1 | 3.28 s | 74.9 s | 23x | 27.9 MB | 9.9 MB |
| 8 | 0.59 s | 10.4 s | 18x | 30.4 MB | 55.4 MB |

1998 genes by 20000 cells (2000 requested), library size 500, 3 552 556 stored
counts (8.9% dense):

| threads | wall, ours | wall, reference | speedup | peak RSS, ours | peak RSS, reference |
|---|---|---|---|---|---|
| 8 | 22.3 s | 415.1 s | 19x | 1079 MB | 937 MB |

Our split at that size: parse 0.26 s, compute 14.7 s, write 7.3 s. A third of
the wall clock is now formatting 40 million numbers as text. The harness keeps
an extra copy for `log_transcription_quotients`, 320 MB of our peak.

The outputs agree to the precision written: per-gene correlation of the log
transcription quotients and of their error bars is 1.0000 on every gene, and
the largest absolute difference in log transcription quotient is `3e-6` at both
sizes, against six decimals in the text files.

## GPU against CPU

Both paths in one process, `examples/profile_gpu.rs`, sanity-sc-rs bed11ca. CPU
is `sanity` on 10 threads in `f64`; GPU is `sanity_gpu` through wgpu on Metal.
Library size 500, 161 bins over `[1e-3, 50]`. Wall clock of the call only,
kernels compiled beforehand. Errors are the GPU against the CPU, worst gene:
the log fold change in units of the CPU's own error bar, and the absolute error
in the log transcription quotient `m + d_c`.

1998 genes by 20000 cells:

| rule | CPU | GPU | speedup | worst `d_c` / `e_c` | worst `m + d_c` |
|---|---|---|---|---|---|
| `Marginalise` | 14.4 s | 0.82 s | 17.7x | `4.0e-4` | `1.1e-4` |
| `PosteriorMean` | 14.1 s | 0.55 s | 25.7x | `4.6e-4` | `1.7e-4` |
| `MaxPosterior` | 14.2 s | 0.61 s | 23.2x | `4.4e-5` | `8.1e-6` |
| `Fixed(1.0)` | 1.53 s | 0.21 s | 7.2x | `5.5e-5` | `1.5e-6` |

500 genes by 200000 cells:

| rule | CPU | GPU | speedup | worst `d_c` / `e_c` | worst `m + d_c` |
|---|---|---|---|---|---|
| `Marginalise` | 63.2 s | 2.36 s | 26.8x | `1.9e-3` | `5.3e-4` |
| `PosteriorMean` | 61.9 s | 2.24 s | 27.6x | `2.1e-3` | `5.3e-4` |
| `MaxPosterior` | 62.9 s | 2.23 s | 28.2x | `2.1e-4` | `5.6e-6` |
| `Fixed(1.0)` | 4.04 s | 0.57 s | 7.0x | `2.7e-4` | `1.5e-6` |

Every error sits below 1% of an error bar, which is as far as the variance grid
itself is resolved. The worst genes are the most expressed ones at large `v`,
where the posterior over the variance is broad and small differences in the bin
likelihoods move the weights. Bonsai reads `log_fold_changes`, so the `d_c`
column is the one that reaches it.

The CPU path agrees with the reference binary to the precision written, so the
GPU's worst `m + d_c` of `5.3e-4` at 200000 cells is also its distance from the
reference.

`MaxPosterior` re-solves bins the device cannot separate from the best in `f64`
on the CPU. Those re-solves run the same sparse path as `sanity`, so they cost
little, and the rule lands on the same bin as the CPU.
