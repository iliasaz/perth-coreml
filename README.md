# PerthCoreML

A Swift + CoreML port of [Perth](https://github.com/resemble-ai/Perth) (Perth-Net Implicit), the
audio watermarker used by Resemble AI's TTS stack. Perth-Net Implicit embeds an inaudible
watermark by running the low 128 bins of a signal's magnitude spectrogram through a small residual
convolutional network and adding the result back before resynthesis; a matching decoder network
reads the same subband and reports a confidence score for whether a clip is watermarked. This
package reimplements that pipeline for iOS and macOS: the encoder and decoder conv stacks run as
CoreML `.mlpackage`s that CoreML's compute plan places on the Apple Neural Engine, while the
STFT/ISTFT, dB normalisation, masking, interpolation, and softmax combine all run on the host in
Accelerate/vDSP fp32 — CoreML has no complex-number support, and none of those operations need the
ANE anyway. The watermarker's native rate is Perth's own 32 kHz; 24 kHz audio is resampled in and
back out with a matched pair of polyphase FIR filters chosen to reproduce what Python's librosa
does on each of the two paths Perth actually uses.

## Install

Swift Package Manager, `swift-tools-version: 6.0`:

```swift
.package(url: "https://github.com/iliasaz/perth-coreml.git", branch: "main")
```

then depend on the `PerthCoreML` library product (and, if you want the command-line tool, on the
`perth-cli` executable product). The package has no external dependencies — only CoreML,
Accelerate, and Foundation.

**Platforms:** iOS 18+ and macOS 15+, arm64 (`Package.swift` declares `.iOS(.v18)` /
`.macOS(.v15)`; Apple silicon is the ANE story, but nothing here rules out an Intel Mac at
`cpuOnly`/`cpuAndGPU`).

## Quick start

```swift
import PerthCoreML

let watermarker = try PerthWatermarker(modelDirectory: modelsURL)   // .all compute units, fp16, .preserveLength

let watermarked = try watermarker.applyWatermark(signal, sampleRate: 24_000)
let score = try watermarker.getWatermark(watermarked, sampleRate: 24_000)   // 1.0
```

`PerthWatermarker` is a plain `final class` — initialization and both calls are synchronous and
`throws`, not `async`. Its full surface:

```swift
public enum LengthPolicy: Sendable {
    case pythonParity      // reproduces Python's truncation, incl. the dropped trailing frame
    case preserveLength    // pads/trims so output.count == input.count. Default.
}

public final class PerthWatermarker {
    public init(modelDirectory: URL,
                computeUnits: MLComputeUnits = .all,
                useFP32: Bool = false,
                lengthPolicy: LengthPolicy = .preserveLength) throws

    public func applyWatermark(_ wav: [Float], sampleRate: Int) throws -> [Float]
    public func getWatermark(_ wav: [Float], sampleRate: Int, round: Bool = true) throws -> Float
}
```

- `modelDirectory` must contain `PerthEncoder.mlpackage` and `PerthDecoder.mlpackage` (or, with
  `useFP32: true`, `PerthEncoder_fp32.mlpackage` / `PerthDecoder_fp32.mlpackage`) — see below for
  how to produce them. If a precompiled `PerthEncoder.mlmodelc` / `PerthDecoder.mlmodelc` is
  already sitting next to the `.mlpackage`, it's loaded directly; otherwise the package is
  compiled once and the result is cached under Application Support, keyed by the package's
  content, so a second launch doesn't pay a fresh ANE compile.
- `computeUnits` and `useFP32` are the two knobs that pick your accuracy/placement tier.
  `.cpuOnly` + `useFP32: true` is the numeric-parity reference tier; `.cpuAndNeuralEngine` or
  `.all` + `useFP32: false` (the default) is the ANE tier — see Accuracy below for what each
  costs you.
- `sampleRate` accepts only `32_000` (Perth's native rate, no resampling) and `24_000` (resampled
  through the built-in polyphase filters); anything else throws `PerthError.unsupportedSampleRate`.
- Only the pure convolutions run in CoreML; everything else (STFT/ISTFT, dB normalise, masking,
  interpolation, the masked mean, softmax, the residual add) runs on the host. See
  `docs/architecture.md` for why the split is drawn there.

## Getting the models

The weights are **not** in this repo, and they are not embedded in the Swift sources either.
(`Sources/PerthCoreML/PerthAssets.swift` does carry base64 blobs, but those are only the 2048-tap
STFT window and the resampler FIR tables — DSP constants, not network weights.) The conv weights
live inside the `.mlpackage`s, which are published to a **private** Hugging Face repo:

**`iliasaz/perth-coreml`** — mirroring how the rest of this stack ships models.

| package | precision | size | needed for |
|---|---|---|---|
| `PerthEncoder.mlpackage` | fp16 | 4.5 MB | `applyWatermark` |
| `PerthDecoder.mlpackage` | fp16 | 13 MB | `getWatermark` only |
| `PerthEncoder_fp32.mlpackage` | fp32 | 9 MB | numeric-parity reference tier |
| `PerthDecoder_fp32.mlpackage` | fp32 | 27 MB | numeric-parity reference tier |

The repo is private, so pulling it needs an HF token with read access to it:

```
hf download iliasaz/perth-coreml --local-dir ./models
perth-cli in.wav --models ./models --cu ane
```

**If you only embed watermarks, you only need `PerthEncoder`.** The decoder is for verification and
QA. `PerthWatermarker.init` currently loads both eagerly, so a caller that never detects is paying
13 MB it doesn't use — worth making lazy if that matters to you.

All four packages declare `Float32` inputs and outputs regardless of tier; the fp16 packages differ
only in `compute_precision` (fp16 weights and internal activations, with implicit casts at the graph
boundary). `docs/architecture.md` has the full I/O contract.

### Rebuilding them from source

The packages are reproducible from a local Perth checkout — nothing about them is hand-tuned:

```
cd converter
python convert_perth_coreml.py --window 1024 --out ../out
```

This traces Perth's encoder and decoder conv stacks with `converter/perth_wrappers.py`'s
static-window wrapper, converts each to an `mlprogram` at `minimum_deployment_target=iOS18`, and
writes the four packages plus `perth_assets.json`. If you change the checkpoint, also re-run
`converter/gen_swift_assets.py` — it regenerates `PerthAssets.swift`, whose STFT window is read out
of the checkpoint itself and is **not** interchangeable with a freshly computed Hann window (see
`docs/architecture.md`).

## Integrating with chatterbox-coreml

**This is done.** [`chatterbox-coreml`](https://github.com/iliasaz/chatterbox-coreml)
watermarks every utterance it generates, on by default, using this package — matching
upstream Python chatterbox, which ends `ChatterboxTTS.generate` with an unconditional
`apply_watermark(...)`. The integration lives in its `Sources/ChatterboxCoreML/Watermarker.swift`;
what follows describes it, and is no longer a set of instructions for a change nobody has made.

**Shape of it.** An `AudioWatermarking` protocol with a deliberately **non-throwing** contract: a
watermarker that cannot mark a buffer returns it unchanged. Watermarking is an obligation of the
product, not a correctness precondition of synthesis, so a 4.5 MB side model failing to load must
never fail a user's generation — it logs `[watermark] UNAVAILABLE` and emits plain audio instead.
The encoder is resolved from an explicit directory, then the chatterbox model directory, then
`iliasaz/perth-coreml` on the Hub; only `PerthEncoder` (4.5 MB) is fetched, since the decoder is
detection-only and this package loads it lazily. The knob is **load-time only** and the app exposes
no switch — a watermark the end user can toggle per utterance is not a watermark.

### The per-chunk question, resolved by measurement

Earlier revisions of this file said flatly: *watermark the whole utterance, not each streamed
chunk*, and warned that if you went the other way you had to measure it. chatterbox-coreml went
the other way — `generateStream` hands each chunk to its caller before the next one exists, so
waiting for a complete utterance would mean streaming audio that is never marked at all — and it
measured it.

The concern is real and worth restating: `magmask` thresholds every frame against the loudest frame
**in the signal it is given** (`Spectral.magmask`, 5 % of peak energy), so a quiet chunk and a loud
chunk are gated against different references, and every chunk boundary is a seam in the residual.
That is a reason to check, not a reason to assume failure.

Measured on chatterbox turbo, three sentences at `--sentence-pause 0.25` (190,010 samples), with
this package's own `perth-cli --detect`:

| signal | detector score |
|---|---|
| whole utterance, watermarked per chunk | **1.0** |
| its three thirds, scored separately | 0.9995 / 1.0 / 1.0 |
| whole utterance, watermarking disabled | **0.0** |

So per-chunk embedding survives detection here, whole and in pieces, with a clean negative control.
It is pinned by a test on that side (`WatermarkEndToEndTests`) rather than left as a comment,
because the failure mode is silent: the audio still plays, it just stops carrying a mark.

**This result is specific to that pipeline's chunk sizes**, which are sentence-scale. It is not a
general licence to watermark arbitrarily short fragments — see the next paragraph for the floor.

### One sharp edge, stated precisely

`applyWatermark` does **not** refuse a signal that is too short to transform. It returns it
**unchanged**: `guard x32.count >= PerthConfig.minSamples32k else { return wav }`
(`Sources/PerthCoreML/PerthWatermarker.swift:51`) — 1025 samples at 32 kHz, ~43 ms at Perth's
native rate, ~32 ms of 24 kHz input. Nothing throws and nothing logs. `getWatermark` is the
opposite: it throws `PerthError.signalTooShort` on the same input
(`PerthWatermarker.swift:119`).

The practical consequence for any chunked caller: a sub-43 ms tail chunk goes out silently
unmarked. That is the right default for embedding — failing a generation over a 30 ms fragment
would be absurd — but it means **a caller cannot infer from the absence of an error that its audio
was marked**. Check the audio, or check the log.

## CLI usage

```
perth-cli <in.wav> --models <dir> [--out <o.wav>] [--detect] [--fp32] \
          [--cu cpu|gpu|ane|all] [--python-parity]
```

- `--models <dir>` (required): the directory from the previous section.
- With no `--detect`, it watermarks `<in.wav>` and, if `--out` is given, writes the result;
  either way it prints the round-trip detection score on its own output as a sanity check.
- `--detect` runs `getWatermark` instead and prints both the raw and banker's-rounded score.
- `--fp32` loads the `_fp32` packages; pair it with `--cu cpu` for the numeric-parity tier.
- `--cu` maps to `MLComputeUnits`: `cpu` → `.cpuOnly`, `gpu` → `.cpuAndGPU`, `ane` →
  `.cpuAndNeuralEngine`, `all` → `.all` (the default).
- `--python-parity` selects `LengthPolicy.pythonParity` instead of the default
  `.preserveLength`, which matters when diffing output against stock Python Perth.

## Tests

`swift test` runs 39 offline unit tests across seven suites — STFT/ISTFT, interpolation, tiling,
the resampler, `magmask`, and score rounding/clipping — asserted against fixtures generated
directly from the Python libraries each piece reproduces (`converter/gen_test_fixtures.py`), not
just against other Swift code. An eighth suite, `End to end`, drives the real `PerthWatermarker`
API through the actual CoreML models; it's skipped unless `PERTH_MODEL_DIR` is set:

```
PERTH_MODEL_DIR=$(pwd)/out swift test
```

## Accuracy

Measured by `converter/validate_swift.py` (Gate 3: the real `perth-cli` binary vs. Python's
`perth.PerthImplicitWatermarker`, on real speech), current as of the last commit:

| input | tier | result |
|---|---|---|
| 32 kHz (native, no resampler) | fp32 / `cpuOnly` | SNR 123.8 dB, max\|err\| 8e-07, cos 0.9999999 |
| 32 kHz (native, no resampler) | fp16 / ANE | cos 0.999994; Python's own detector scores the Swift-watermarked audio **1.000000** |
| 24 kHz (the chatterbox path, resampler in the loop) | fp32 / `cpuOnly` | SNR 107–120 dB |
| 24 kHz (the chatterbox path, resampler in the loop) | fp16 / ANE | cos 0.999991 |

Cross-detection passes both ways on every clip in the corpus (Python's detector accepts
Swift-watermarked audio, and vice versa), and output lengths match exactly under
`.pythonParity`.

### On device, and on the Neural Engine

The same comparison run on an iPhone 17 Pro Max (iOS 26.5.1, Release build) by `repro/PerthProbe`,
over five seconds of speech:

| input | tier | apply | result |
|---|---|---|---|
| 32 kHz | fp32 / `cpuOnly` | 16 ms | cos 1.0, SNR 123.8 dB, detect 1.0 |
| 32 kHz | fp16 / ANE | **8 ms** | cos 0.9999936, detect 0.9993 → 1 |
| 24 kHz | fp32 / `cpuOnly` | 65 ms | SNR 107.5 dB, detect 1.0 |
| 24 kHz | fp16 / ANE | 50 ms | cos 0.9999934, detect 0.9993 → 1 |

The fp32 tier reproduces the Mac's 123.8 dB exactly on device. There is no iPhone-versus-Mac
Neural Engine numeric surprise here, which is not something to take for granted — it holds because
these are plain convolutions, with no fused-attention kernel whose behaviour can differ between the
two ANE generations.

**Neural Engine residency is proven, not inferred.** Asking for `.cpuAndNeuralEngine` is only a
hint; CoreML is free to ignore it. An Instruments Core ML trace records ANE *hardware intervals*,
and both models show `Prediction` intervals on the Neural Engine — a model that compiled for the
ANE but silently fell back to CPU or GPU would show a `Load` interval with **zero** predictions:

| model | ANE load | ANE prediction |
|---|---|---|
| `PerthEncoder` | 4.1 ms | **0.3 ms** |
| `PerthDecoder` | 14.9 ms | **1.2 ms** |

So the encoder — the only model `applyWatermark` needs — watermarks five seconds of audio in
0.3 ms of Neural Engine time. Reproduce with `xcrun xctrace record --template "Core ML"` against
`repro/PerthProbe`; see `docs/architecture.md`.

**Literal byte-for-byte identity with stock Python Perth is not achievable, and that's expected,
not a bug.** PyTorch links Sleef for its transcendental math (`log10`, `atan2`, `cos`/`sin`, `pow`);
Apple links libm. vDSP's FFT sums the same numbers in a different radix decomposition than
PyTorch's pocketfft. Every one of those differences is sub-ULP to a few ULP individually, but
they compound across an STFT → conv stack → ISTFT round trip. The fp32-on-CPU tier above is
therefore "numeric parity at the fp32 noise floor" — 107 to 124 dB SNR against Python, cosine
similarity indistinguishable from 1.0 — not bitwise equality. The fp16-on-ANE tier trades a bit
more of that margin for Neural Engine placement and still clears cosine similarity ≥ 0.99999
against Python on every measured clip.

## Gotchas

- **`applyWatermark` can return fewer samples than you gave it — unless you use the default
  `LengthPolicy`.** Perth's ISTFT is called without an explicit `length:`, so it drops the
  trailing partial frame; under `.pythonParity` that's up to `hop - 1` samples short (up to 239
  samples / ~10 ms at 24 kHz). `.preserveLength`, the default, pads before the STFT and trims
  back afterward so you get exactly the length you passed in — use `.pythonParity` only when
  diffing against Python.
- **`getWatermark` returns NaN on digitally silent audio.** The confidence score is a masked
  mean, and silence makes the mask all-zero, so the denominator is zero. Stock Perth does the
  same thing (`0/0` in numpy) — this isn't a bug we introduced, it's Perth's own contract.
- **Very short signals can't be watermarked.** The STFT reflect-pads by `nFFT/2 = 1024` samples,
  which needs at least that many real samples to mirror. Below 1025 samples at 32 kHz,
  `applyWatermark` returns the input unchanged (it does not throw); `getWatermark` throws
  `PerthError.signalTooShort` instead, since there's no sensible score to return.
