# Architecture

This document explains how `perth-coreml` is put together and why the boundaries are drawn where
they are. It describes the code as it exists in `Sources/PerthCoreML` and `converter/`, not the
original design exploration in `docs/porting-plan.md` (which records the reasoning and the
rejected alternatives, and is worth reading if you want the "why not" as well as the "why").

## The host / CoreML split

Only the pure `Conv1d` + `LeakyReLU` stacks run in CoreML. Everything else — the STFT and ISTFT,
dB (de)normalisation, the semantic mask (`magmask`), both flavors of `F.interpolate`, the masked
mean, the softmax combine, and the residual add — runs on the host in Accelerate/vDSP, always in
fp32 (`Sources/PerthCoreML/STFT.swift`, `Spectral.swift`, `Interpolate.swift`,
`PerthWatermarker.swift`).

The reason is not performance, it's representability: CoreML's `mlprogram` format has no
complex-number type. Perth's STFT produces complex spectra, `cx_to_magphase` needs `atan2` and
`hypot`, and reassembling a signal on the way out needs `cos`/`sin` of the phase and a genuine
overlap-add — none of that traces into a CoreML graph. So the boundary is drawn at the one place
that's already real-valued and small: the dB-normalised magnitude subband, `sub_mag` /
`residual`, both `(1, 128, W)`. That tensor's range is bounded and unremarkable — Perth's own
normalisation (`(20·log10(mag) + 180) / 195`) keeps it in roughly `[0, 1.23]` by construction —
so it survives fp16 conversion with room to spare; the `1e-9 … 1e10` linear-magnitude domain the
STFT actually works in never crosses into a CoreML tensor.

Each conv stack — the encoder, and each of the decoder's three branches (`slow`/`norm`/`fast`) —
is architecturally identical: an initial `Conv1d(128→256, k=1)` with a bias, five
`Conv1d(256→256, k=7, padding=3)` layers, and a final `Conv1d(256→C_out, k=1)` that emits either
the encoder's 128-channel residual or the decoder's 2-channel `(attention logit, watermark
estimate)` pair. `LeakyReLU` follows every layer but the last. `converter/check_ane.py`, run
against the built packages, confirms this op-for-op: `PerthEncoder.mlpackage` has exactly 7
`conv`, 6 `leaky_relu`, and 8 `mul` (the validity-mask multiply applied once before the first
layer and once after each of the 7 layers); `PerthDecoder.mlpackage` has exactly 3× that (21
`conv`, 18 `leaky_relu`, 24 `mul`) for its three branches.

## The static-window + validity-mask design

This is the part that makes the CoreML port possible at all, and the easiest part to get subtly
wrong.

The ANE needs a fixed input shape at compile time, but Perth's conv stacks are meant to run over
a signal of arbitrary length. The five `k=7, padding=3` layers give a receptive field of
`1 + 5·(7-1) = 31` frames, so cutting the signal into windows and running each one independently
would corrupt `halo = 15` frames at each cut — the standard "chunk with overlap and discard the
edges" story.

**The naive fix — zero-pad the window and let PyTorch's own boundary handling take over — is
wrong, and it's wrong in a way that's easy to miss because it doesn't crash or look obviously
broken.** `Conv1d(128, 256, k=1)` carries a bias, so `LeakyReLU(conv(zeros))` is not zero — a
zero-padded tail "lights up" after the very first layer, and that spurious activation then leaks
backward into the last real frames at every subsequent layer through the k=7 convolutions.
Measured in `docs/porting-plan.md`: an all-zero input to the layer stack produces activations
with `absmax ≈ 0.0225`, against a genuine watermark residual whose own magnitude peaks around
`0.017` — **the zero-padding artifact is larger than the entire watermark**, and it contaminates
exactly the last `halo` frames of every window regardless of how large you make the halo, because
it's a boundary-bias error, not a receptive-field error.

The fix that is actually correct — and the one shipped, in both the PyTorch reference
(`converter/perth_wrappers.py`) and the Swift runtime (`Sources/PerthCoreML/PerthWatermarker.swift`)
— is to zero the activations, not just the input, after every layer:

```python
h = x * vmask
for layer in stack:
    h = layer(h) * vmask
```

This is what PyTorch's own `padding=3` already does at every layer when you run the full-length
signal through the unmodified model: it zero-pads *activations*, not just the raw input, at every
convolution. Reproducing that with an explicit mask multiply after each layer makes a
zero-padded, fixed-width window produce **exactly** the same output as running the whole signal
through PyTorch at full length — not approximately, bit-for-bit. That claim is Gate 1
(`converter/validate_decomposition.py`), and it's checked directly: `apply()`'s output matches
stock Perth with `max|err| == 0.0` across every tested window size and signal length.

The window width shipped is `W = 1024` frames (`PerthConfig.modelWindow` in Swift,
`window_frames` in `perth_assets.json`, and the `--window` flag to
`converter/convert_perth_coreml.py` — note the script's own CLI default is `512`; the shipped
packages were built with `--window 1024` explicitly, and the Swift side hard-codes 1024, so
regenerating the models needs that flag passed by hand). With `halo = 15`, the tiler
(`converter/perth_wrappers.py:tile_plan`, mirrored exactly in `Sources/PerthCoreML/Tiling.swift`)
walks the signal in a stride of `W - 2·halo = 994` frames: an interior window's first and last 15
outputs are discarded (they were computed from a fake cut edge), while the very first window's
left edge and the very last window's right edge are true signal boundaries where the model's own
padding-plus-mask is already correct, so nothing is discarded there. Coverage is gap-free and
overlap-free by construction — the next window's first *owned* output is exactly where the
previous window's ownership ends — and a signal shorter than `W` collapses to a single window
with the mask handling the difference, with no separate short-signal code path. The overlap cost
is `2·15 / 1024 ≈ 2.9%` of redundant compute per window.

## `vmask` is not `magmask`

These are two different masks with similar shapes and completely different meanings, and
confusing them is, in the porting-plan's own words, the single easiest bug to write in this
project.

- **`vmask`** (`Tiling.swift` / the `window(...)` helper in `PerthWatermarker.swift`) is a
  bookkeeping device for the tiler. It's `1.0` on frames that are genuinely part of the current
  window and `0.0` on the zero-padded tail — **including on real frames that happen to be
  silent**. It exists purely to keep a fixed-width CoreML window numerically identical to running
  the full-length signal through the unmodified conv stack, and it is applied *inside* the CoreML
  graph, after every layer.
- **`magmask`** (`Spectral.swift` on the Swift side, `magmask()` in `converter/perth_pipeline.py`)
  is semantic, not structural: it's `1.0` on frames whose total energy across **all 1025** STFT
  bins exceeds 5% of the loudest frame's energy in the whole (untiled) signal, and `0.0`
  otherwise — the "is there actually something here to watermark or detect" gate. It never enters
  the CoreML graph at all (the model only ever sees 128 of the 1025 bins, so it structurally
  can't compute this), and it operates on the *global*, full-length signal, computed once on the
  host before tiling begins.

Feeding `magmask` where `vmask` belongs would zero out legitimate activations for quiet-but-real
frames inside a window, changing what the model computes rather than just neutralising padding.
Feeding `vmask` where `magmask` belongs would fail to gate quiet frames out of the watermark
add / detector's masked mean, changing the watermarker's semantics. Neither mistake crashes or
produces a shape error — both are silent numeric corruption.

`magmask`'s own implementation has a smaller, quieter trap worth knowing about: it accumulates
each frame's 1025-bin energy sum in `Double` (`Spectral.swift`), then computes the threshold as
`Float(peak) * Float(0.05)` — **not** promoted to `Double` before multiplying — because that's
what matches PyTorch's own fp32 arithmetic. The comparison against the threshold is a strict `>`.

## The CoreML model contract

Two packages, each shipped in an fp16-compute and an fp32-compute variant, built by
`converter/convert_perth_coreml.py` and measured directly from the artifacts in this repo:

| package | compute precision | declared I/O dtype | inputs | output | size on disk |
|---|---|---|---|---|---|
| `PerthEncoder.mlpackage` | fp16 (`compute_precision=FLOAT16`) | `Float32` | `sub_mag` `(1,128,1024)`, `mask` `(1,1,1024)` | `residual` `(1,128,1024)` | 4.7 MB |
| `PerthEncoder_fp32.mlpackage` | fp32 | `Float32` | same | same | 9.5 MB |
| `PerthDecoder.mlpackage` | fp16 | `Float32` | `slow_x`/`slow_m`, `norm_x`/`norm_m`, `fast_x`/`fast_m` — each `(1,128,1024)` / `(1,1,1024)` | `slow_out`/`norm_out`/`fast_out` — each `(1,2,1024)`: channel 0 the branch's attention logit, channel 1 its watermark estimate | 14.0 MB |
| `PerthDecoder_fp32.mlpackage` | fp32 | `Float32` | same | same | 28.0 MB |

Note that **all four packages declare `Float32` inputs and outputs**, verified directly against
each package's `MLModelDescription` — `compute_precision` only controls the weights and internal
activations inside the `mlprogram`, not the declared boundary dtype. The "fp16" packages carry
implicit fp32↔fp16 casts at their input and output (visible as the `cast` ops in
`check_ane.py`'s output: 3 per encoder-shaped stack, 9 for the decoder's three branches).
Deployment target is `iOS18` (`minimum_deployment_target=ct.target.iOS18`); conversion emits
`convert_to="mlprogram"`.

On this Mac, `converter/check_ane.py` — which compiles each package with `xcrun coremlcompiler`
and reads back `MLComputePlan`'s per-op device preference — shows all 24 of the encoder's
non-constant ops preferred on the Neural Engine, with every op also supporting CPU and GPU. The
decoder's 72 ops, despite being structurally identical (three copies of the same 7-conv stack),
are all preferred on the GPU instead, though every one of them supports the Neural Engine too;
that's CoreML's own placement heuristic on this hardware; it is not a limitation of the graph.
This is a Mac-side, compile-time signal — `check_ane.py`'s own docstring calls it exactly that,
a smoke test — not proof of what a given iPhone actually does at runtime.

## Numeric traps

Each of these was found by measuring the actual disagreement, not by inspecting the math and
assuming it would be fine. Each is called out in the source at the point where it matters.

- **The STFT window comes from the checkpoint, not from a freshly computed Hann window.**
  Perth's `AudioProcessor` registers its window as an `nn.Module` buffer, so
  `load_state_dict(...)` overwrites torchaudio's default with whatever
  `perth_net_250000.pth.tar` actually stored. That stored window differs from
  `torch.hann_window(2048)` by one ULP (`5.96e-08`) — and using a fresh window instead puts a
  roughly `2e-7` floor under every subsequent waveform comparison. The 2048 taps are exported
  once by `converter/gen_swift_assets.py` and baked into `PerthAssets.stftWindow` as a base64
  blob; Swift never recomputes it.
- **Interpolation index arithmetic must be `Float32`, not `Double`.** `F.interpolate`'s linear
  and nearest modes compute a source index from a scale factor, and PyTorch does that arithmetic
  in the tensor's own dtype. `Sources/PerthCoreML/Interpolate.swift` measured, against torch
  2.8.0: doing the index math in `Double` instead of `Float` lands nearest-mode on a different
  source index in 571 of 1668 tested `(T, S)` pairs (34%); for linear mode it produces an actual
  off-by-one, `max|Δ| = 3.05e-5`. "More precise" is not "more correct" here.
- **The linear interpolation blend needs an explicit `fma`.** Swift does not auto-contract
  `a*b + c*d` the way ATen's fused kernel does; the non-contracted form disagrees with torch in
  over half of sampled cases. `Interpolate.swift` calls `.addingProduct(_:_:)` explicitly rather
  than relying on the compiler.
- **The ISTFT's window-square envelope is not constant, and dividing by its average is a real
  bug, not a simplification.** `2048 / 320 = 6.4` is not an integer, so `hann²` does not satisfy
  the constant-overlap-add condition. `STFT.swift` accumulates the true envelope per sample and
  divides by it; the code and the commit history both note that dividing by the constant
  `Σw²/hop` instead would put roughly a 29% amplitude error into the first and last 1024 samples
  of every utterance.
- **`torch.round` is banker's rounding; the obvious Swift equivalents are not.** Perth's detector
  score is rounded with round-half-to-even (`0.5 → 0.0`, `2.5 → 2.0`); Swift's default
  `Float.rounded()` is round-half-away-from-zero. `PerthWatermarker.getWatermark` uses
  `.rounded(.toNearestOrEven)` specifically.
- **The clip before rounding must propagate NaN, and the obvious fast paths silently don't.**
  `PerthWatermarker.getWatermark` clips the raw score with `Swift.min(Swift.max(score, 0), 1)`
  rather than `Float.minimum`/`Float.maximum` or `simd_clamp` — both of the latter turn a NaN
  input into `0.0`, which would make digitally silent audio silently report "not watermarked"
  instead of correctly reporting "no answer" (see the NaN-on-silence behavior in Gotchas in the
  README).
- **There are two resampler tap sets, because Python's Perth uses two.**
  `apply_watermark` resamples with librosa's default `res_type`, which is `soxr_hq`; `get_watermark`
  resamples with `res_type="polyphase"`, i.e. SciPy's `resample_poly`. This package does not
  vendor libsoxr (it's LGPL), so `converter/gen_swift_assets.py` designs an 801-tap
  Kaiser(β=13.75) `firwin` at soxr's own cutoff as a stand-in for the apply path — measured to
  reproduce soxr_hq to roughly 112 dB SNR against librosa, versus roughly 54 dB if
  `resample_poly`'s filter were used instead. Using the higher-quality apply-path filter on the
  *detect* path would be a bug, not an improvement: its different stopband shifts `magmask`'s
  threshold (the dB transform amplifies stopband differences summed across all 1025 bins) and
  flips mask bits, and the actual gate is "our detector agrees with Python's detector," which
  uses `resample_poly`. `Resampler.swift` keeps the two filter pairs (`hqUp24to32`/`hqDown32to24`
  vs `polyUp24to32`) strictly on their respective paths.
- **librosa pads with a zero sample when the 24→32 kHz ratio doesn't divide evenly.** soxr's
  natural output length is `floor(n·4/3)`; librosa's `fix_length` pads up to `ceil(n·4/3)` by
  appending a literal zero whenever `n % 3 == 1` — a third of all possible input lengths.
  `Resampler.up24to32Apply` reproduces that by forcing the last output sample to zero under the
  same condition, rather than letting the FIR compute whatever value falls out of the filter.
- **`AVAudioFile.read(into:)` does not promise to fill the buffer in one call.** On a plain
  float32 WAV, a single `read(into:)` returned 159,724 of 160,000 requested frames — silently
  shortening the signal and shifting every downstream frame count. `PerthCLI.swift`'s `readWav`
  loops until the file is exhausted rather than trusting one call.

## The validation ladder

Three gates, each isolating a different layer of the port from the ones above it. All three run
against `perth.PerthImplicitWatermarker` (stock Python Perth) as ground truth.

| Gate | What it validates | Script | fp32 threshold | fp16/ANE threshold |
|---|---|---|---|---|
| 1 | The windowed/tiled conv stack, still executing in **PyTorch** (no CoreML involved) — isolates "did we decompose the model correctly" from "did CoreML convert it faithfully." | `converter/validate_decomposition.py` | `apply()` matches stock Perth with `max\|err\| == 0.0`, exactly, across window sizes {256, 512, 1024} and six signal lengths | `detect()` agrees with stock Perth to `\|Δ\| < 1e-5` pre-rounding, and the rounded decision is identical (both-NaN counts as agreement) |
| 2 | The **saved `.mlpackage` files** (not the in-memory conversion) driven through the real host pipeline, across `cpuOnly` and `ALL` compute units | `converter/validate_coreml.py` | apply `max\|err\| ≤ 1e-5`; rounded detect decision matches; cross-detection passes both ways (each side's detector accepts the other's watermarked audio) | `cos ≥ 0.9999`; `\|ΔSNR\| ≤ 0.3` dB vs Python; rounded detect decision matches; cross-detection passes both ways |
| 3 | The **compiled `perth-cli` binary** against stock Python, on real speech, at native 32 kHz and at 24 kHz (resampler in the loop) | `converter/validate_swift.py` | output length matches Python's exactly; `SNR ≥ 90` dB; rounded cross-detection matches | output length matches Python's exactly; `cos ≥ 0.9999`; rounded cross-detection matches |

Gate 2's own numbers, measured against the built packages: fp32/`cpuOnly` apply reaches
`max|err| = 2.7e-07` (the fp32 noise floor); fp16/`ALL` reaches `cos = 0.999997` with identical
rounded decisions on both sides. Gate 3's numbers — the ones that matter for a real caller, since
they exercise the actual Swift host DSP and the actual CLI — are in the README's Accuracy
section. Gate 2 deliberately checks **both** `cpuOnly` and `ALL`, not just one, because the two
diverge (fp16 residual RMS differs between them) and a bug can hide if only one is reported.

`converter/validate_decomposition.py`'s test signal (`make_signal`) is synthetic — tones plus
noise, with a leading silence — which is sufficient for the DSP-parity checks in Gates 1 and 2,
where the point is exercising the numeric pipeline under controlled conditions. Gate 3 uses real
speech specifically because detector-score gates are meaningless on synthetic tones: Perth's own
detector can score an *unwatermarked* tone-plus-noise clip high enough to round to "watermarked."
