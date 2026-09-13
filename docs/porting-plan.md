# Porting Perth (Implicit) to Swift + CoreML — Implementation Plan

Status: plan of record. Synthesized from five deep dives (`stft`, `coreml`, `decoder`, `resampler`, `integration`) and their adversarial critiques. Where a dive and its critique conflict, §11 records the ruling and the reason.

---

## 0. Architecture of record (one paragraph)

Host (Swift/Accelerate, **fp32, always**) owns: resampling, STFT, `cx_to_magphase` (abs → clip 1e-9 → 20·log10 → normalize), `magmask` (all 1025 bins), the residual add, both `F.interpolate` variants, the masked mean, the softmax, `magphase_to_cx`, ISTFT (true overlap-add envelope, no COLA shortcut). CoreML owns **only the pure Conv1d+LeakyReLU stacks**, at a **single static shape `W = 1024` frames**, driven by a host **halo tiler** with a **validity mask (`vmask`) applied after every layer** — the one construction that is *bit-exact* (`torch.equal == True`) against the full-length PyTorch conv stack and simultaneously handles `T < W` with no second shape and no fallback path. CoreML has no complex-number support and `torch.angle` / `exp(1j·φ)` will not convert, so the boundary is drawn at the normalized-dB magspec subband `(1,128,·)` and the residual `(1,128,·)` — both O(1), safely inside fp16 range; the `1e-9`…`1e10` linear-magnitude domain never crosses into a CoreML tensor.

---

## 1. Two exactness tiers — restated with measurable thresholds

### 1.1 The literal "byte match" gate is unattainable. Read this before Phase 0 sign-off.

This is the one place the plan pushes back on a stated decision, and it does so with numbers, not opinion:

| stage | why bitwise is impossible | measured floor |
|---|---|---|
| STFT/ISTFT | torch uses pocketfft; vDSP uses a different radix decomposition and summation order. torch's own fp32 STFT already differs from fp64 by 1.4e-7 rel-to-peak. | max\|Δ\| / peak\|F\| = **1.34e-7**; only **1.93 %** of complex bins bitwise identical |
| dB / phase math | torch links **Sleef**; Apple links **libm/vForce**. Bitwise agreement torch-vs-numpy fp32, 1 M samples: `log10` 99.60 %, `10**x` 98.04 %, `atan2` 94.31 %, `cos` 80.89 % | not bitwise |
| CoreML fp32 conv | different accumulation order than PyTorch's kernel | max\|Δresidual\| = **9.7e-8** (rel 5e-6) |
| masked-mean reduction | ATen's blocked/vectorized fp32 reduction matches neither naive, pairwise, nor double | \|Δscore\| ≤ **2.4e-7** |

What *is* genuinely bit-exact and **stays a hard bitwise gate**:
- the **halo/vmask tiling** vs full-length PyTorch (`torch.equal == True`, verified twice independently, with non-zero biases);
- the **Hann window** (shipped as a blob, see §5.1);
- **`magmask`** bit-vector (0/1 per frame);
- **linear interp** (`fp32` scale + `fmaf`): 400/400 and 12/12 at C=128 vs torch;
- **nearest interp** indices (`fp32` scale): 1668/1668 (T,S) pairs;
- **normalize/denormalize** (`(db+180)/195`, `·195−180` — both constants exact in fp32).

### 1.2 Tier-1 — "fp32 numeric parity" (replaces byte-match)

Every threshold below is met with ≥10× margin by measurements already taken. Path: host fp32 + **`PerthEncoderF32`/`PerthDecoderF32`**, `MLComputeUnits.cpuOnly`.

| # | Gate | Threshold | Measured |
|---|---|---|---|
| T1.1 | tiler vs full-length PyTorch conv stack | **`torch.equal` == True** | ✅ exact |
| T1.2 | STFT complex vs torch | `max\|Δ\| / peak\|F\| ≤ 1e-6` | 1.34e-7 |
| T1.3 | ISTFT waveform vs torch (same spec) | `SNR ≥ 110 dB` | 140.5 dB |
| T1.4 | `magmask` bit-vector | **identical on every corpus clip** | 0 flips / 432 slices |
| T1.5 | interp (linear + nearest) | **bit-exact** | ✅ |
| T1.6 | CoreML-fp32 residual vs torch | `max\|Δ\| ≤ 1e-6` | 9.7e-8 |
| T1.7 | e2e audio vs Python `apply_watermark` | `cos ≥ 1 − 1e-7` **and** `SNR ≥ 110 dB` | 1.0000001 / 130.1 dB |
| T1.8 | `\|ΔSNR(orig,wm)\|` vs Python | `≤ 0.01 dB` | 4.4e-7 dB |
| T1.9 | output length | **exactly equal** to Python's | see §4 |
| T1.10 | detector, **pre-clip** scalar | `\|Δ\| ≤ 1e-4` and rounded bit identical, all clips | 2.4e-7 |

**Gates are worst-case over the corpus (§2), not mean, not single-clip.**

### 1.3 Tier-2 — "fp16 / ANE functional equivalence"

Path: host fp32 + **`PerthEncoderF16`/`PerthDecoderF16`**, `MLComputeUnits.cpuAndNeuralEngine` **and** `.all`, on **iPhone 17 Pro Max**.

| # | Gate | Threshold |
|---|---|---|
| T2.1 | `cos(swift_ane_wm, python_wm)` | **≥ 0.9999**, worst clip |
| T2.2 | `\|ΔSNR(orig,wm)\|` vs Python | **≤ 0.1 dB**, worst clip |
| T2.3 | Python detector on Swift-ANE-watermarked audio | rounds to **1** on every clip; pre-clip ≥ 0.9 |
| T2.4 | Swift detector vs Python detector, same input | **rounded bit identical** on every clip (clean + watermarked) |
| T2.5 | ANE residency | 100 % of non-const ops on ANE, confirmed by **Instruments trace on device** (not just `MLComputePlan`) |
| T2.6 | fp16 pathology sentinel | **no exactly-zero output tensors**, no NaN |
| T2.7 | device ≠ Mac sanity | `cos(device_ane, mac_cpu_fp32)` on the same fixture ≥ 0.9999 |

**STOP-LOSS (new, and load-bearing):** if the **Mac** ANE worst-case `|ΔSNR|` exceeds **0.05 dB** (half the budget) in Phase 1, do **not** spend device cycles on the plain fp16 encoder — go straight to the hybrid split (§8, R1 fallback b). Rationale: the sibling project inflated Mac→iPhone ANE error by ~100×; there is no scenario where a Mac number at 0.085 dB survives on device.

---

## 2. The validation corpus (built once, Phase 0, used by every gate)

Single most common failure mode across all five dives: **one clip, presented as proof.** Three dives report Mac-fp16-ANE `ΔSNR` as −0.019 dB, −0.036 dB, and −0.085 dB. They cannot all be right. All gates are corpus-worst-case.

`fixtures/corpus/` — ≥30 clips, mono, committed as 16-bit wav + `manifest.json`:
- **20 real chatterbox-coreml TTS outputs at 24 kHz** (the actual input distribution — generate them, don't guess).
- 5 real speech clips (different speakers, incl. leading/trailing digital silence and inter-sentence pauses).
- Adversarial: full-scale (peak 0.999); **high-passed > 2.5 kHz** (the case where the 0–2 kHz subband sits at the fp32 noise floor — known-degraded, see §9); **whole-file digital silence**; silence head + fade-to-zero tail; impulse; DC; white noise.
- Length coverage: `N ∈ {1025, 1026, 32000, 32001, 32319, 32320, 47999, 24001 (n%3==1), 143998, 100000}` — i.e. every residue class that the length contract and the resampler care about.

> **Do not validate detection on synthetic tones.** Measured: an unwatermarked tone+noise clip scores **0.9507 → rounds to 1** in stock Python; and Python's own watermark on a sine+noise clip scores **0.335 → rounds to 0** after an STFT/ISTFT round trip. Real speech only for detector gates; tones are for DSP-parity gates only.

---

## 3. The CoreML contract (exact)

### 3.1 Tiling / vmask — the whole ANE story

**Receptive field** = `1 + 5·(7−1) = 31` frames ⇒ **halo H = 15** on each side (verified: perturb input frame 250 → outputs [235, 265] change, width 31).

**The lead's sketch is wrong as written and must not be built.** "Boundary chunks use zero padding, matching PyTorch's `padding=3`" is false: layer 0 is `Conv1d(128,256,k=1)` **with a bias**, followed by LeakyReLU, so a zero input frame becomes `LeakyReLU(b₀) ≠ 0`. Measured: `layers(all-zero input).absmax = 0.0225`, versus a watermark residual of `|max| = 0.017`. **The padding artifact is larger than the entire watermark**, contaminates exactly the last 15 frames of every utterance (`max|Δ| = 3.06e-2`), and is **invariant to halo size** — it is a signal-boundary error, not a receptive-field error. Anyone tuning the halo will burn days.

**The fix that is bit-exact — multiply by a validity mask after every layer, inside the CoreML graph:**

```python
h = x * vmask
for layer in stack:
    h = layer.conv(h)
    if layer.act: h = F.leaky_relu(h, 0.01)
    h = h * vmask
```

*Proof (induction).* For `t < L`, `conv(x_pad)[t] == conv(x_full)[t]`: taps at index ≥ L read exactly 0 in `x_pad`, and PyTorch's own boundary padding supplies 0 in `x_full`; taps < 0 are zero in both. LeakyReLU is pointwise. The trailing `* vmask` re-zeroes `[L, W)`, restoring the invariant for the next layer. ∎ Empirically `torch.equal(masked_pad_output, full_length_output) == True`, reproduced independently by two agents including with non-zero biases.

**`vmask` is NOT `magmask`.** `vmask` = 1 on real frames, 0 on pad — *including silent real frames*. `magmask` is 0 on quiet interior frames and would change the model if used here. This is the easiest bug in the project to write.

**The tiler (`W = 1024`, `H = 15`, stride `S = W − 2H = 994`):**

```
s = 0
loop:
    L  = min(W, T - s)                       // real frames in this window
    x_win[0..<L] = mag_sub[..., s..<s+L];  x_win[L..<W] = 0
    vm[0..<L]    = 1;                      vm[L..<W]    = 0
    o  = coreml(x_win, vm)                   // (1, C_out, W)
    lo = (s == 0)     ? 0 : H                // first window owns its left edge
    hi = (s + W >= T) ? L : W - H            // last  window owns its right edge
    out[..., s+lo ..< s+hi] = o[..., lo..<hi]
    if s + W >= T { break }
    s += S
```

Coverage is gap-free and overlap-free by construction: the next window's claimed start is `s + S + H = s + W − H`, which is exactly this window's `hi`. A non-final window guarantees `L' ≥ W − S + 1 = 2H + 1 = 31 > H`, so `hi > lo` always. `T ≤ W` ⇒ **one window**, `lo = 0`, `hi = L = T`, fully correct via the vmask — **no short-signal fallback path exists or is needed**. (Do not write `nwin = max(1, ceil((T−W)/994)+1)`; integer ceil-div truncates toward zero for a negative numerator. The loop above is the spec.)

Overlap cost: `30/1024 = 2.9 %`. `W = 1024` was the best measured per-frame point (0.418 µs/frame; 1536 → 1.08, 2048 → 0.95) and equals **10.24 s of audio per predict**. A 60 s utterance = 7 encoder windows ≈ 3 ms on ANE.

**Rejected alternative — edge-anchored, no-explicit-pad tiling** (also bit-exact, `max_abs_err = 0.0`): it requires `T ≥ W`, and for the decoder the binding constraint is `floor(3T/4) ≥ W` — at `W = 128` that is 1.70 s of audio, and `T_fast` bottoms out at 3 for the shortest legal input. There is no fixed `W` that covers all `T`, so it forces a second static shape or a CPU fallback below the threshold. The vmask design makes that entire branch of the plan disappear.

### 3.2 Decoder tiling

`magmask` (global max over the **true, unpadded** `T`, over **all 1025 bins**) and both interpolations are **global** ops. Compute them **host-side, at full length, before tiling**. Then tile the conv stack over the already-interpolated array. The masked mean is **linear**, so accumulate `Σ(x·m)` and `Σm` across windows and divide once.

```
T_slow = 5*T/4      // Int division. NOT Int(Double(T)*1.25).
T_fast = 3*T/4      // verified identical to int(T*0.75) for all T in [0, 200000]
for branch b in {slow: T_slow, normal: T, fast: T_fast}:
    x_b   = lerp(mag_sub[0..<128, 0..<T], T -> T_b)     // host, per-channel, fp32, see §5.4
    m_b   = nearest(mask, T -> T_b)                     // host, fp32-scale floor, see §5.4
    numA, numW, den = 0.0 (Double)
    tile x_b with (W=1024, H=15); over each claimed range [lo,hi):
        for t in lo..<hi:
            g = Double(m_b[s+t])
            numA += Double(o[0][t]) * g                 // channel 0 = attn
            numW += Double(o[1][t]) * g                 // channel 1 = wmark
            den  += g
    attn_b = Float(numA/den); wm_b = Float(numW/den)    // den == 0 -> §9 NaN policy
```
`normal` branch: `T_b == T` ⇒ both interps are the identity (no special case needed; both formulas already degenerate correctly).

### 3.3 Package / function / tensor contract

| package | precision | function(s) | inputs | output | size |
|---|---|---|---|---|---|
| `PerthEncoderF16.mlpackage` | fp16 weights, **fp16 I/O**, `compute_precision=FLOAT16` | `main` | `x`: `(1,128,1024)` fp16<br>`vmask`: `(1,1,1024)` fp16 | `res`: `(1,128,1024)` fp16 | 4.74 MB |
| `PerthEncoderF32.mlpackage` | fp32, fp32 I/O, `compute_precision=FLOAT32` | `main` | same, fp32 | same, fp32 | 9.4 MB |
| `PerthDecoderF16.mlpackage` | fp16, fp16 I/O | **multifunction**: `slow`, `normal`, `fast` | `x`: `(1,128,1024)`<br>`vmask`: `(1,1,1024)` | `out`: `(1,2,1024)` (ch0=attn, ch1=wmark) | 14.0 MB |
| `PerthDecoderF32.mlpackage` | fp32 | same 3 functions | same | same | ~28 MB |

- `minimum_deployment_target = ct.target.iOS18` (matches chatterbox-coreml's macOS 15 / iOS 18 floor; iOS26 buys nothing).
- **`ct.TensorType`, never `ImageType`.**
- **No `RangeDim`, no `EnumeratedShapes`, no quantization.** Static shape is the entire ANE strategy.
- `compute_units` at convert time is a hint only; the real choice is `MLModelConfiguration.computeUnits` at load.
- **Multifunction buys one file, not one load.** `MLModelConfiguration.functionName` binds one `MLModel` to one function ⇒ **three `MLModel` instances, three AOT compiles** (measured 197/136/130 ms on Mac). It also **does not dedup** (18,746,983 B vs 18,748,108 B summed — 1,125 bytes saved). It is kept purely so the compiled-model cache manages one `.mlmodelc` key instead of three. Do not claim any other benefit.
- Chatterbox only calls `apply` ⇒ **the decoder packages are lazily downloaded**, never pulled by a caller that only watermarks.

Converter recipe (`converter/export_encoder.py`, verbatim shape):

```python
class Stack(nn.Module):
    def __init__(self, seq): super().__init__(); self.seq = seq
    def forward(self, x, vmask):
        h = x * vmask
        for l in self.seq:
            h = l.conv(h)
            if l.act: h = F.leaky_relu(h, 0.01)
            h = h * vmask
        return h

W = 1024
ts = torch.jit.trace(Stack(net.encoder.layers).eval(),
                     (torch.randn(1,128,W), torch.ones(1,1,W)))
m = ct.convert(ts,
    inputs =[ct.TensorType(name="x",     shape=(1,128,W), dtype=np.float16),
             ct.TensorType(name="vmask", shape=(1,1,  W), dtype=np.float16)],
    outputs=[ct.TensorType(name="res", dtype=np.float16)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18)
m.save("PerthEncoderF16.mlpackage")
```

`conv1d` needs **no** conv2d rewrite: coremltools emits rank-3 `ios18.conv` natively, Espresso canonicalizes to 4-D itself, and a hand-written `(1,C,1,W)` version produces **byte-identical output at identical latency** (0.44 vs 0.45 ms).

---

## 4. The length contract (a real integration bug — decide it before code lands)

`AudioProcessor.magphase_to_signal` calls `InverseSpectrogram` with **no `length=`** and `center=True`. Therefore:

```
out32 = 320 * (T - 1) = floor(N32 / 320) * 320          // up to 319 samples SHORTER than the input
out24 = ceil( floor(ceil(N24 * 4/3) / 320) * 320 * 3/4 ) // up to 239 samples (9.96 ms) shorter
```
Measured: `32001→32000`, `32319→32000`, `47999→47680`, `24001→24000`, `12345→12240`, `100000→99840`. Max drop at 24 kHz = **exactly 239 samples** (at `n24 = 959`), brute-forced over `n ∈ [769, 400000]`.

**Minimum input:** `N32 ≥ 1025` (reflect-pad limit), i.e. `N24 ≥ 769`. Below that Python **throws**. Swift must guard and return the input unmodified rather than crash.

Consequence: `N32 ≥ 1025 ⇒ T ≥ 4 ⇒ T_slow ≥ 5, T_fast ≥ 3`. The disputed `T ≥ 2` / `T ≥ 3` decoder guards are **unreachable via the audio path** — but implement the `D == 1 ⇒ scale = 0` interp case anyway (it is what ATen's `area_pixel_compute_scale` does), so the code is correct if the conv stack is ever called directly.

**Two policies, both shipped:**

| policy | behaviour | used by |
|---|---|---|
| `.pythonParity` | reproduce the truncation exactly | **all Tier-1/Tier-2 gates** — the `cos ≥ 0.9999 vs Python` gate is meaningless unless lengths match |
| `.preserveLength` | pad the 32 kHz signal up to a multiple of 320 → watermark → trim to `ceil(N24·4/3)` → resample down → fix to `N24` | **default for the chatterbox integration API** |

`.preserveLength` verified: returns exactly the caller's length; Python detector still fires (1.000000); the recovered tail is genuinely watermarked (`detect(last 0.5 s alone) = 0.998284` vs stock's own tail 0.998598); agrees with stock on the common prefix at 45–52 dB SNR (differs only near the seam, where the extra STFT frame changes the overlap-add).

---

## 5. Host DSP — the exact recipes

### 5.1 STFT

- `T = N // 320 + 1`. Output `(1025, T)` complex64.
- Reflect pad 1024 each side, **no edge repeat**: `xp[i] = x[1024−i]` for `i<1024`; `xp[1024+N+i] = x[N−2−i]`.
- **Every original sample enters the STFT for all `N ≥ 1025`.** The claim "319 trailing samples never enter the STFT" is **false** (`N − 320·(N//320) ≤ 319 < 1024`, so the last original sample always sits inside a frame; exhaustively checked `N ∈ [1025, 60000]`). Only the ISTFT *output* is truncated. Do not "fix" a non-existent information loss.
- **Ship `torch.hann_window(2048)` as an 8192-byte fp32 blob** (`Resources/hann2048.f32`). Recomputing it in Swift with `cosf` yields **62/2048 taps off by 1 ULP** (Sleef vs Apple libm), and ATen's fp32 op order catastrophically cancels near n=0 (`w[1] = 2.3543835e-06` vs the fp64-exact `2.35309521e-06` — a **5.5e-4 relative** error). **Never use `vDSP_hann_window`** — it produces the *symmetric* window (`max|Δ| = 1.18e-3` vs periodic). Do not "correct" the blob with an fp64-accurate window; that silently diverges.
- vDSP packing, **measured on this machine, do not trust memory**:
  - `vDSP_fft_zrip` forward output is **exactly 2× the true DFT**. Multiply by 0.5.
  - **Nyquist's real part lives in `imagp[0]`**, not in a 1025th slot. DC in `realp[0]`. Both have zero imaginary part.
  - Inverse: pack the **true** (already ×0.5) spectrum → `zrip` INVERSE → `ztoc` → **× 1/2048** (not 1/4096).
  - `FFTSetup` created once, shared read-only; `DSPSplitComplex` buffers are **per-thread**.

### 5.2 `cx_to_magphase`

`mag = hypot(re,im)` → `max(mag, 1e-9)` (**before** log10, or digital silence → `-inf` → NaN through the net) → `db = 20·log10(mag)` → `norm = (db + 180) / 195`.
Constants are exact in fp32: `20·log10(1e-9) == −180.0`, `(180 + 15) == 195.0`. Digital silence ⇒ normalized magspec **exactly 0.0**.
Phase = `atan2f(im, re)`. **`vvatan2f(z, y, x, n) → atan2(y, x)` — imaginary part is the SECOND argument.**

**Range fact (state the bound, do not sample it):** with `|x| ≤ 1` and `Σw = 1024`, `|X| ≤ 1024` ⇒ `db ≤ 60.2` ⇒ `norm ≤ (60.206 + 180)/195 = 1.232`. Measured max 1.157. The claim "normalized magspec range ≈ [0,1]" is **wrong** (it exceeds 1), but harmlessly so — the domain is logarithmic, so even `|x| = 100` only reaches 1.44. Never near fp16's 65504.

### 5.3 `magphase_to_cx` + ISTFT

```
db  = norm * 195 - 180
e   = min(db / 20, 10)                     // torch: .clip(max=10)
lin = 10 ** e                              // vvpowf(z, y, x, n) => z = x**y : EXPONENT 2nd, BASE 3rd
re  = lin * cos(phase); im = lin * sin(phase)   // vvsincosf(sin, cos, x, n) : SIN first
```
ISTFT (exactly ATen's recipe; an fp64 reimplementation matches `torch.istft` to 2.7e-7):
```
L   = 2048 + 320*(T-1)
frames = irfft(spec, 2048) * (1/2048)     // DC/Nyquist imaginary parts are DISCARDED — verified Δ = 0.0
frames *= window                           // window applied a SECOND time
acc  = OLA(frames, hop=320) over L
env  = OLA(window², hop=320) over L
y[n] = acc[1024+n] / env[1024+n]  for n in 0 ..< 320*(T-1)      // NO eps, NO constant
```
**The envelope is NOT constant.** `2048/320 = 6.4` is not an integer ⇒ hann² does not satisfy COLA. Measured over the *kept* region: `env[0] = 1.700312`, `env[512] = 2.392405`, steady-state `[2.399397, 2.400624]`. **Dividing by the constant `Σw²/hop = 2.4` gives a 29.15 % amplitude error in the first and last 1024 samples of every utterance.** Just OLA `w²` — it is `T × 2048` adds. NOLA min is 1.700 for every `T ∈ [4, 101]` ⇒ torch never raises; add no eps.

**Silence must stay exactly silent**, and it does so *only because `hann[0] == 0`*: silence → magspec 0 → −180 dB → `lin = 1e-9` in every bin → `irfft` = a delta at sample 0 of each frame → multiplied by `w[0] = 0` → exactly zero. A Swift ISTFT that windows in a different order, or uses a shifted window, turns digital silence into 1e-9 noise and breaks the gate. **This is a required unit test.**

### 5.4 Interpolation (torch-exact — fp32 index math is load-bearing)

```swift
// linear, align_corners=true
let scale: Float = (S > 1) ? (Float(T - 1) / Float(S - 1)) : 0      // Float32 divide. D==1 -> 0.
for d in 0..<S {
    let r: Float = scale * Float(d)                                 // Float32 multiply
    let i0 = min(Int(r.rounded(.down)), T - 1)
    let i1 = i0 + (i0 < T - 1 ? 1 : 0)
    let lam: Float = r - Float(i0)
    out[d] = fmaf(x[i0], 1 - lam, x[i1] * lam)   // == (x[i1]*lam).addingProduct(x[i0], 1-lam)
}

// nearest
if S == T { identity }
let ns: Float = Float(T) / Float(S)                                 // Float32, NOT Double
idx[d] = min(Int((Float(d) * ns).rounded(.down)), T - 1)
```
- **Double-precision index math is WRONG.** linear: `max|Δ| = 3.05e-5` (an actual off-by-one). nearest: **571/1668 mismatches (34 %)** — reproduced in Swift: `T=30, S=22, d=11` → fp32 gives 15, fp64 gives 14, torch gives 15. Also `S=254, D=190, i=95` → torch 126, integer `(i·S)/D` 127.
- **Swift does not auto-contract `a*b + c*d`.** You must call `fmaf` explicitly; the non-FMA form fails (1225/2800 bitwise, `max|Δ| = 4.77e-7`).
- ⚠️ The winning FMA form (`fmaf(x0, w0, x1*lam)`) contradicts a plain reading of ATen's source (which predicts the *other* contraction, and which measures 2/400). **This is an artifact of a specific vectorized kernel + compiler.** ⇒ **Pin `torch==2.8.0` in `converter/`, and re-run the 3000-case sweep on any torch upgrade.** Treat "bit-exact" as "bit-exact vs torch 2.8.0 arm64 CPU".

### 5.5 `magmask`

```swift
// s over ALL 1025 bins (the CoreML model only sees 128 — this CANNOT be moved into the graph)
let s   = (0..<T).map { t in Float(sum over k in 0..<1025 of Double(magspec[k][t])) }   // Double accumulate
let thr = s.max()! * Float(0.05)     // Float(0.05).bitPattern == 0x3d4ccccd — cast to fp32, NOT promoted to Double
let mask = s.map { $0 > thr ? Float(1) : Float(0) }   // STRICT >
```
`torch(smax * 0.05) == Float(smax) * Float(0.05)` in **20000/20000** trials; the Double-promoted form matches only 15988/20000.

Encoder computes the mask on the **pre-residual** magspec (it clones, masks, then `+= res`). Decoder uses `.detach()` (numerically identical).

**Fragility:** `s ≈ 666…852` for real audio vs `thr ≈ 42.6`; min `|s − thr|` across 432 real slices = **6.98e6 ULP**, 0 flips under ±2-ULP perturbation. But on band-limited audio, `|s_torch − s_swift|` reaches **1.67e-1**, and swapping the *resampler* alone flipped **1 mask bit in 651 frames**. ⇒ **CI alarm:** log `min_t|s_t − thr| / max_t|Δs_t|` over the corpus; **fail the build if the ratio ever drops below 100×** (current worst observed: 624×).

### 5.6 Softmax, masked-mean, clip, round

```swift
let m  = max(a0, max(a1, a2))                          // torch uses max-subtraction
let e0 = expf(a0-m), e1 = expf(a1-m), e2 = expf(a2-m)
let sum = (e0 + e1) + e2                               // left-to-right fp32
let wmark = ((w0 * e0/sum) + (w1 * e1/sum)) + (w2 * e2/sum)   // left-to-right fp32 (bit-exact 5000/5000)
let score = Swift.min(Swift.max(wmark, 0), 1)          // NOT simd_clamp / Float.minimum
let out   = round ? score.rounded(.toNearestOrEven) : score   // BANKER'S rounding
```
- **NaN-clamp trap (measured in Swift):** `Swift.min(Swift.max(nan,0),1) → nan` ✅ (matches torch); `vDSP_vclip → nan` ✅; **`Float.minimum(Float.maximum(nan,0),1) → 0.0` ❌**; **`simd_clamp(nan,0,1) → 0.0` ❌**.
- **`Float.rounded()` is WRONG** (`0.5 → 1.0`); torch uses banker's rounding (`0.5 → 0.0`, `2.5 → 2.0`). Use `.rounded(.toNearestOrEven)`. Saturated 0.0/1.0 values are *common* after the clip (raw watermarked score = **1.00034**, i.e. the `.clip(0,1)` is load-bearing).
- Masked-mean numerator in **Double** (`vDSP_dotprD`), denominator as an exact `Int` count. Torch's own fp32 reduction is *less* accurate than double accumulation; the resulting `|Δscore|` vs torch is ≤ 2.4e-7.

### 5.7 Resampler (4:3 / 3:4 polyphase, Accelerate)

**Two prototypes, selected by path** — this is a correction to the resampler dive:

| path | Python reference | Swift filter | measured agreement |
|---|---|---|---|
| `applyWatermark` 24↔32 | librosa default = **`soxr_hq`** | **`hq`**: 801-tap, `firwin(801, 11482/48000, ('kaiser', 13.75))` | `cos = 1.000000000`, SNR **101.6 dB** (up) / **97.0 dB** (down) |
| `detectWatermark` 24→32 | `res_type="polyphase"` = **scipy `resample_poly`** | **`scipy81`**: `firwin(81, 0.25, ('kaiser', 5.0)) × up` | worst `max\|Δ\| = 2.4e-7` (≤2.5 ULP) |

Using `hq` on the detect path (as the resampler dive recommended) is wrong: it measurably shifts the `magmask` threshold (43.6 → 38.3 on watermarked speech, because the dB transform amplifies stopband differences across all 1025 bins) and **flips mask bits**. The gate is *"Swift detector agrees with Python detector"*, and Python's detector uses `polyphase`. Detect-path filter quality is irrelevant (the detector's clean-vs-watermarked margin is 370×). **Match Python on both legs.**

Engine (`vDSP_desamp` per residue class — each output phase is exactly a decimating FIR):
```
p     = down*m + halfLen ;  phase = p % up ;  base = p / up
y[m]  = Σ_i h[phase + up*i] * x[base - i]        // h scaled by `up`; x zero outside [0,n)
n_out = ceil(n * up / down) = (n*up + down - 1) / down       // matches scipy AND librosa, all n
```
Constants: `hq` halfLen = 400; 24→32 (`up=4,down=3`) `Lp=201`; 32→24 (`up=3,down=4`) `Lp=267`. Pad `Lp + down` zeros each side.

**Two bugs to fix in the prototype at `swift-resampler/`:**
1. **librosa's `fix_length` appends a ZERO.** soxr's natural output is `floor(n·up/down)`; librosa pads to `ceil(...)` **with a zero** when `down ∤ n·up` — that is **1/3 of all input lengths** for 24→32 (`n24 % 3 == 1`). Measured `n24 = 24001`: `|last-sample Δ| = 2.38e-3`, `SNR drops to 68.9 dB`, `python_last == +0.0`. **Under `.pythonParity`, force `y[last] = 0` on the 24→32 leg when `(3 ∤ 4·n24)`.** The 32→24 leg in `apply` can never hit it (`out32` is a multiple of 320 and `4 | 320`). Add `n % 3 == 1` lengths to the cross-validation harness — they are currently absent from every test.
2. **Double-rounding of the taps.** Swift does `Float(prototype[i] * Double(up))`; scipy does `h = firwin(...).astype(x.dtype)` **then** `h *= up`. For `up = 3` (the 32→24 direction) **22/81 taps differ by 1 ULP**. Round to `Float` **first**, then multiply.

Also: the source doc-comment claims "bit-compatible with `scipy.signal.resample_poly`" — it is not (≤2.5 ULP). Correct the comment. `cblas_scopy` is deprecated: compile with `-DACCELERATE_NEW_LAPACK`.

**CI guard (S11):** `converter/extract_soxr_kernel.py` re-derives the 801 taps from the pinned `soxr==1.1.0` and asserts `max|Δ| ≤ 1e-5` against the committed table. Without it, a soxr upgrade silently moves the parity baseline.

---

## 6. Phases and exit gates

> Ordering principle: cheapest disconfirming experiment first. **But Phase 2 (device) runs before the bulk of the Swift host DSP is written**, because it is the only gate that can invalidate the architecture, and it needs *nothing* but the `.mlpackage`s + Python-dumped `.npy` fixtures. Everything after it is device-independent host code.

### Phase 0 — Ground truth, corpus, gate sign-off (Python only; ~1 day)

| Experiment | Kill / decision criterion |
|---|---|
| **E0.1** Build `fixtures/corpus/` (§2). Generate 20 real 24 kHz chatterbox TTS outputs. Record the **distribution of `T`**. | — |
| **E0.2** Run stock Perth over the corpus; dump `orig`, `python_wm`, output **lengths**, `SNR(orig,wm)`, `cos(orig,wm)`, **pre-clip** detector scores (clean + watermarked). | Establishes every reference number. If Python's own `SNR(orig,wm)` is not ~14–16 dB and `cos(orig,wm)` ~0.985, the checkpoint is not what we think it is. |
| **E0.3** Present §1.1 to the user and get explicit sign-off replacing "byte match" with the Tier-1 table. | **This is a blocking decision, not a proposal.** Literal bit-equality is provably unattainable (Sleef≠libm, pocketfft≠vDSP, CoreML conv≠ATen conv). |
| **E0.4** *(time-boxed, 2 h, optional)* Vendor `pocketfft_hdronly.h`, compile arm64, feed it the same windowed frames, count bitwise-identical bins vs torch. | If **100 % bitwise**, we keep a genuine byte-match gate on the STFT stage and add a C++ target. If **< 100 %**, drop it immediately and do not add the C++ target. Verified prerequisite: this torch build links pocketfft (`pocketfft::detail::general_nd` present; `DftiComputeForward` absent). |
| **E0.5** Confirm where chatterbox applies the watermark: **whole concatenated utterance**, not per streamed chunk. | Per-chunk is **not** equivalent: measured `cos(whole, per-chunk) = 0.999975`, `max|Δ| = 0.0103` vs a watermark amplitude of 0.0133 — **the seam error is 77 % of the watermark**. Three mechanisms break it (per-chunk reflect padding, per-chunk ISTFT truncation, per-chunk `magmask` global max). "Stateless convs ⇒ per-chunk is also exact" is **false**. If chatterbox *requires* mid-stream watermarking, the whole plan's parity gates are measuring the wrong object and the design must change. |

**EXIT GATE 0:** corpus + reference `.npy`/`.json` committed; Tier-1 table signed off; whole-utterance hook confirmed; `T` distribution known.

### Phase 1 — Converter + Mac coremltools parity (Python; ~2 days)

| Experiment | Kill criterion |
|---|---|
| **E1.1** vmask exactness sweep: `torch.equal(tiled, full)` for `T ∈ {4, 100, 1023, 1024, 1025, 2000, 3000}` × encoder + 3 decoder branches, in **PyTorch**. | Any `False` ⇒ the vmask proof is wrong ⇒ fall back to edge-anchored tiling + a `T < 4W/3` CPU fallback (§11, ruling 1). |
| **E1.2** Build all four `.mlpackage`s at the **exact shipping config** (fp16 I/O for F16!). Verify with `MLComputePlan` **and** by reading the spec that the shipped artifact has the dtypes we claim. | The previously-measured "100 % ANE, 3 casts" was taken on an **fp32-I/O** build that is *not* the shipping artifact. **Never interpret a number without verifying what was actually loaded.** |
| **E1.3** Tier-1 gates T1.6/T1.7/T1.8/T1.9/T1.10 on `PerthEncoderF32` + `PerthDecoderF32`, `CPU_ONLY`, over the **full corpus**, worst case. | `max\|Δresidual\| > 1e-6` ⇒ the CoreML fp32 conv is not the parity oracle we assumed ⇒ fall back to a BNNS/Accelerate fp32 conv stack on the host for Tier-1. |
| **E1.4** Tier-2 accuracy on `PerthEncoderF16` (fp16 I/O), **corpus worst case**, on `cpuOnly` / `cpuAndGPU` / `cpuAndNeuralEngine` / `all`. Report `cos`, `ΔSNR`, residual rel-RMS, and **check for exactly-zero tensors**. | **STOP-LOSS: Mac-ANE worst `\|ΔSNR\| > 0.05 dB` ⇒ do NOT go to device with plain fp16.** Jump straight to R1-fallback-(b) (hybrid split, §8). The three existing measurements disagree 4× (−0.019 / −0.036 / −0.085 dB); this experiment is the tiebreak and it is *the* decision point of the project. |
| **E1.5** Decoder tiling through CoreML (not just PyTorch): score vs torch on the corpus. | `\|Δscore\| > 1e-5` (fp32) or rounded-bit disagreement (fp16) ⇒ the masked-mean accumulation across windows is wrong. |
| **E1.6** Numpy-fp32 reference host DSP built literally from §5 (STFT, dB, magmask, interp+FMA, masked-mean, softmax) vs torch on the corpus. Dump every intermediate as `.npy`. | These `.npy` files become the **oracles for Phase 3**. Already validated on 6 inputs at `max\|Δ\| = 2.4e-7`; extend to the corpus. |
| **E1.7** Mac load-time: fresh process per row, `PerthEncoderF16` cold vs warm, `cpuOnly` vs `.all`. | Known Mac numbers: **75 ms (cpuOnly) / 158 ms (.all)** warm. ANE costs **+84 ms of load to save 0.85 ms per 10 s window** — it breaks even after ~100 utterances per process. Feed this into the Phase 5 default-CU decision. |

**EXIT GATE 1:** four `.mlpackage`s built at shipping config; Tier-1 fp32 gates all green on the corpus; Mac-ANE Tier-2 worst-case `|ΔSNR| ≤ 0.05 dB` and `cos ≥ 0.9999`; `MLComputePlan` shows 100 % ANE on the fp16 packages; host-DSP `.npy` oracles committed.

### Phase 2 — iPhone ANE probe (device; ~1 day) — **the architecture-deciding gate**

Nothing but the `.mlpackage`s and `.npy` fixtures is needed. `repro/PerthProbe-iOS`: one minimal SwiftUI app (`RuaccentProbe-iOS` shape, not the sprawling `ANEProbe` shape). **`os.Logger` + `idevicesyslog -u <UDID> --process PerthProbe`** — the brief's "os.Logger never reaches `--console`" learning is **stale**: chatterbox's `CLAUDE.md` (corrected 2026-05-29) documents that `idevicesyslog` *does* stream `.notice`+`.debug` on iOS 26.5, and every current probe logs exclusively via `os.Logger`. (`devicectl --console` only relays stdout — that's the real constraint.)

| Experiment | Kill criterion |
|---|---|
| **E2.1** `PerthEncoderF16`, **one (model × compute-unit) per process** (a failed `MLModel` load SIGBUSes uncatchably inside CoreML's AOT compiler). `cpuOnly` → `cpuAndGPU` → `cpuAndNeuralEngine` → `all`. Compare against the Mac-CPU-fp32 `.npy` baseline. | `cos(device, mac_cpu_fp32) < 0.999` on any CU ⇒ **device miscompile** ⇒ R1 fallbacks. **Exactly-zero outputs** ⇒ fp16 pathology (look for zero, not NaN). |
| **E2.2** Full Tier-2 gate on device: reconstruct audio *in Python* from the device-returned residual, compute `cos` and `ΔSNR` vs `python_wm` on the corpus. | `cos < 0.9999` or `\|ΔSNR\| > 0.1 dB` on **any** clip ⇒ plain fp16/ANE is dead. This is the number the sibling project's 0.9999→0.9828 inflation predicts might fail. |
| **E2.3** ANE residency on device via **Instruments + `coreml-trace-analyzer`**. `MLComputePlan` is a *compile-time plan on a Mac*, not proof of device execution. | Any conv or the `vmask` broadcast-`mul` falling to GPU/CPU ⇒ R3 fallbacks. |
| **E2.4** Cold vs warm device load with the `.mlmodelc` at a **stable path** (App Support, `isExcludedFromBackup`), vs `cpuOnly` load as the baseline. | Feeds the default-CU decision. If ANE cold-compile is multi-second, ANE ships as a proven-but-non-default artifact. |
| **E2.5** Same sweep for `PerthDecoderF16` (3 functions). | Decoder ANE is nice-to-have; the hard user criterion is "**the model** runs on the ANE", satisfied by the encoder. Do not block on the decoder. |

**EXIT GATE 2:** T2.1–T2.7 green on iPhone 17 Pro Max for `PerthEncoderF16` on `.cpuAndNeuralEngine` **and** `.all`, with ANE residency confirmed by a device trace, and no exactly-zero/NaN outputs. **If this gate fails, stop and re-plan against §8-R1 before writing any more Swift.**

### Phase 3 — Swift host DSP (macOS CLI, no CoreML; ~4 days)

Build `Sources/PerthCoreML/{STFT, AudioProcessor, Interpolate, PolyphaseResampler, TileScheduler}.swift` and a `PerthCLI dump` subcommand that emits every intermediate as `.npy`. Validate against the Phase-1 oracles.

| Experiment | Kill criterion |
|---|---|
| **E3.1** Empirically verify `vvpowf` / `vvsincosf` / `vvatan2f` argument orders in Swift **before** using them. (`vForce.h`: `vvpowf(z,y,x,n) → x**y`; `vvsincosf(sin,cos,x,n)`; `vvatan2f(z,y,x,n) → atan2(y,x)`.) | Header comments are evidence, not proof. A reversed `vvpowf` is silent and catastrophic. **The entire dB/phase Accelerate stage is currently unmeasured** — the existing prototype's dB numbers were computed in *numpy*, not Swift. |
| **E3.2** T1.2 / T1.3 / T1.4 / T1.5 / T1.9 over the corpus. Including `N = 32001, 32319, 47999` (the existing prototype `stft.swift:94-95` emits `N` samples, not `320*(T−1)` — **it ships the exact bug its own report lists as trap #7**, and every test to date used `N = 32000`, a multiple of 320, which hides it). | Any gate red. |
| **E3.3** Silence test: `applyWatermark(zeros)` returns **exactly** all-zeros (0 nonzero samples). | Non-zero ⇒ the ISTFT windowing order is wrong (§5.3 — silence survives only because `hann[0] == 0`). |
| **E3.4** Resampler: `hq` vs `librosa soxr_hq` at `n24 % 3 ∈ {0,1,2}` (the `%3==1` case is **untested today**); `scipy81` vs `scipy.signal.resample_poly` for `up ∈ {3,4}` (the 1-ULP tap bug is in the `up=3` direction). Round-trip 24→32→24 SNR ≈ 44.2 dB (inherent, not our bug — do not chase it). | `cos < 0.999999` after the `fix_length`-zero fix. |
| **E3.5** Tiler exactness in Swift: run the conv stack (fp32 CoreML, CPU) whole vs tiled; assert **byte-identical**. | Any diff ⇒ the tiler bookkeeping (`lo`/`hi`/stride) is wrong. |
| **E3.6** `magmask` safety-ratio alarm over the corpus. | ratio < 100× ⇒ investigate before shipping. |

**EXIT GATE 3:** all Tier-1 host-DSP gates green on the corpus, worst-case, including the band-limited adversarial clip (documented-degraded, §9) and `N ∉ 320ℤ`.

### Phase 4 — Swift package end-to-end on Mac (~2 days)

Wire `PerthWatermarker` (§7). Port `CompiledModelCache` and `ModelRepository` verbatim from chatterbox-coreml.

| Experiment | Kill criterion |
|---|---|
| **E4.1** Full Tier-1 (fp32/`cpuOnly`) and Tier-2 (fp16/`.all`, `.cpuAndNeuralEngine`) on Mac, corpus worst-case, through the **real Swift API**, `.pythonParity` length policy. | Any gate red. **Validate on BOTH `cpuOnly` and `.all` — they diverge (fp16 residual rms 8.1e-5 vs 2.4e-4) and a bug hides if you report only one.** |
| **E4.2** `detectWatermark` vs Python on the corpus: **pre-clip** scalar `\|Δ\| ≤ 1e-4`, rounded bit 100 % agreement, clean + watermarked. | — |
| **E4.3** 24 kHz round trip, `.preserveLength`: Python detector detects the Swift-watermarked 24 kHz audio (`round → 1`); `cos(swift_24k, python_24k) ≥ 0.999`; **output length == `N24`**. | Decision-#2 gate. |
| **E4.4** Throughput: 60 s utterance, host STFT/ISTFT + resampler + 7 encoder windows. The existing prototype allocates **two Swift `Array`s per frame** — fix before measuring. | No hard gate; report the number. |
| **E4.5** `CompiledModelCacheTests` (fake-`.mlpackage` fixture trick, ported). | — |

**EXIT GATE 4:** `swift test` green offline; `PERTH_MODEL_DIR=… swift test` green with all Tier-1 + Tier-2 Mac gates; 24 kHz round trip functionally validated.

### Phase 5 — Device end-to-end + chatterbox integration (~2 days)

- **E5.1** Run the full Swift pipeline (host DSP + CoreML) on the iPhone via the probe app on 5 corpus clips; compare against Mac `.npy`. **The Accelerate host path has never been run on iOS** — `vDSP_desamp` / `vDSP_fft_zrip` micro-kernels dispatch per core/OS and are not contractually bit-identical Mac↔iPhone. This is a 20-minute test; do not assert "it will behave identically".
- **E5.2** Choose the shipping default CU from E2.4 + E1.7 data. **Decision rule:** if device ANE cold-compile (with a warm `.mlmodelc` at a stable path) is **≤ 200 ms**, ship `.cpuAndNeuralEngine` + fp16 as the default. Otherwise ship fp32/`cpuOnly` as the default and keep the ANE package as the *proven* artifact behind a flag. Be honest: **ANE here is a stated success criterion, not a performance need** (0.43 ms vs 1.31 ms per 10.24 s window; both are noise next to the host STFT).
- **E5.3** Add `perth-coreml` to chatterbox-coreml as a remote SPM dependency; hook `applyWatermark` into `ChatterboxCoreMLModel.generate()` on the **fully concatenated** `[Float]` immediately before `AudioOutput.pcmBuffer(from:)`, with `.preserveLength`. Optional (`watermarker: PerthWatermarker?`, default `nil`), matching the `computeUnits:`/`russianStress:` "new capability defaults to current behavior" precedent.
- **E5.4** Flag to the user: `generateStream` consumers get **unwatermarked** chunks under this design. Per-chunk watermarking is *not* equivalent (E0.5).

**EXIT GATE 5:** Tier-2 green on device end-to-end through the Swift API; chatterbox `generate()` produces audio that stock Python Perth detects (`round → 1`) with correct length; default CU chosen from data.

---

## 7. Public Swift API

```swift
public actor PerthWatermarker {

    public enum Precision: Sendable { case float32, float16 }          // float16 => ANE tier
    public enum LengthPolicy: Sendable {
        case pythonParity      // returns floor(N/320)*320 @32k (up to 319 samples shorter). For gates.
        case preserveLength    // pad→watermark→trim. Returns exactly N. DEFAULT for callers.
    }
    public struct ComputeUnits: Sendable {
        public var encoder: MLComputeUnits = .cpuAndNeuralEngine
        public var decoder: MLComputeUnits = .cpuAndNeuralEngine
        public init() {}
    }

    /// Resolves models from a local directory (.mlpackage or .mlmodelc, either accepted).
    public static func load(from modelDirectory: URL,
                            precision: Precision = .float16,
                            computeUnits: ComputeUnits = .init()) async throws -> PerthWatermarker

    /// Downloads from the private HF repo via swift-transformers `Hub`, then `load(from:)`.
    public static func downloadAndLoad(precision: Precision = .float16,
                                       computeUnits: ComputeUnits = .init(),
                                       progress: (@Sendable (Double) -> Void)? = nil) async throws -> PerthWatermarker

    /// Embeds the watermark. `signal` is mono. Resampled internally to 32 kHz if needed.
    /// Throws `PerthError.signalTooShort` if the 32 kHz signal would be < 1025 samples.
    public func applyWatermark(_ signal: [Float],
                               sampleRate: Int,
                               lengthPolicy: LengthPolicy = .preserveLength) async throws -> [Float]

    /// ONE scalar for the WHOLE utterance (not per-window, not a per-sample confidence).
    /// `round: true` (Python's default) => exactly 0.0 or 1.0, banker's-rounded.
    /// Returns 0.0 for digitally-silent input (Python returns NaN; see PerthError / .nanOnSilence).
    public func detectWatermark(_ signal: [Float],
                                sampleRate: Int,
                                round: Bool = true) async throws -> Float
}

public enum PerthError: Error, Sendable {
    case signalTooShort(samples: Int, minimum: Int)   // n32 < 1025 (n24 < 769)
    case modelNotFound(String)
    case invalidModelOutput(String)
    case unsupportedSampleRate(Int)                   // only 32000 and 24000 have validated resamplers
}
```

- **`actor`, not `final class @unchecked Sendable`.** The chatterbox runners are `@unchecked Sendable` because they are driven by a single dedicated task by construction. `PerthWatermarker` will be called from arbitrary sites. Watermarking runs once per utterance — actor-hop cost is irrelevant. (Note: cross-actor calls are `async`; the integration report's `samples = try watermarker.applyWatermark(...)` **does not compile** — it needs `await`.)
- Parity-mode NaN: expose `detectWatermarkRaw(_:sampleRate:) -> Float` (internal/`@_spi`) that returns `Float.nan` on all-silent input, so the parity harness can compare NaN-for-NaN. The public `detectWatermark` guards `maskSum == 0 → 0.0`.

---

## 8. Top 8 risks — mitigation and named fallback

| # | Risk | Mitigation (before code) | Named fallback |
|---|---|---|---|
| **R1** | **iPhone ANE fp16 fails `cos ≥ 0.9999` / `ΔSNR ≤ 0.1 dB`.** CoreML/ANE accumulates in **fp16** across 1792 MACs/output while the encoder maps an O(1) input (magspec 0.49–1.16) to a **tiny output** (residual `\|max\| = 0.017`) — a ~60× shrink ⇒ catastrophic cancellation. CoreML-fp16 is **35× worse than torch's `.half()`** (0.877 % vs 0.075 %). Mac ANE already sits at **−0.019 to −0.085 dB** of a 0.1 dB budget; the sibling project inflated Mac→iPhone ANE error ~100×. | E1.4 (Mac corpus worst case, **stop-loss at 0.05 dB**) → E2.2 (device). Measure **before** writing `TileScheduler`/`Resampler`/`ModelRepository`. | (a) **Do not** try recentering/rescaling — folding a scale into the k=1 convs is *dead on arrival*: float is scale-invariant, input quantization contributes only 0.002 %, and scaling cannot fix cancellation. (b) **Hybrid split:** CoreML/ANE emits the **penultimate `(1,256,W)` fp16 activations**; the host does the final `Conv(256,128,k=1,act=False)` in **fp32** via `cblas_sgemm` (128×256 GEMM per frame ≈ 98 M MACs for a 60 s utterance ≈ 1 ms). This removes the cancellation site from fp16 and still satisfies "runs on the ANE". Cost: 2× output tensor. (c) GPU-fp16. (d) **CPU-fp32 CoreML** (1.31 ms/window — already fast enough), ANE demoted to a proven-but-non-default artifact with the user informed. |
| **R2** | **`vmask` broadcast-`mul` fragments the graph on device ANE** (Mac says 100 % ANE, but Mac ≠ iPhone). | E2.3: Instruments trace via `coreml-trace-analyzer`, not `MLComputePlan`. | Bake `vmask` as a full `(1,256,W)` / `(1,128,W)` tensor (no broadcast) — 8× the input bytes, still trivial. Or revert to edge-anchored tiling + a `T < 4W/3` CPU fallback (loses the single-shape property, gains nothing else). |
| **R3** | **The "byte match" gate as literally worded is unattainable** (§1.1), so the project could stall on an impossible criterion. | E0.3: get explicit sign-off on the Tier-1 table **in Phase 0**, with the measured evidence. E0.4: the pocketfft probe is the only route that could preserve a genuine bitwise STFT gate. | Ship the Tier-1 numeric table. Keep bitwise gates where they *are* achievable (tiling, window blob, magmask, interp, normalize) — that is not a consolation prize, it covers every op where a bug would be silent. |
| **R4** | **Host DSP silent-parity bugs.** The dB/phase Accelerate stage (`vDSP_zvabs` → clip → `vvlog10f` → `vvpowf` → `vvsincosf`) **has never been written or measured in Swift** — the existing "verified" numbers were computed in numpy. `vvpowf`'s arg order is **reversed** vs C's `powf`. The prototype STFT ships the wrong output length. `vDSP_hann_window` is the wrong window. Dividing by the constant 2.4 gives a **29 % amplitude error**. | E3.1/E3.2: per-op unit tests against Phase-1 `.npy` oracles, before any e2e test. Ship the Hann blob. | None needed — these are deterministic and fully diagnosable. Budget the time. |
| **R5** | **Length-contract truncation ships into `generate()`**, silently shortening every utterance by up to 13 ms and misaligning every `cos` comparison. | §4 `LengthPolicy`; `.preserveLength` verified (correct length, still detects at 1.0, tail genuinely watermarked). Every parity harness aligns on the **shorter** length before computing `cos`. | None. This is a decision, not a risk, once written down. |
| **R6** | **`magmask` bit flip.** It is the only discontinuous operator, sums **all 1025 bins**, and the dB transform amplifies tiny high-band differences. A resampler swap alone flipped 1 bit / 651 frames. On band-limited audio `\|Δs\| = 1.67e-1`. | Double accumulation; `Float(0.05)` (not Double-promoted); strict `>`. **CI alarm** on the safety ratio (fail < 100×; current worst 624×). Match Python's resampler on **both** paths (§5.7). | If a flip is ever observed on the real corpus, gate the detector on the pre-clip score with a documented tolerance band rather than the rounded bit, and escalate. |
| **R7** | **fp16 pathology misdiagnosed as the silence NaN** (or vice versa). fp16 overflow presents as **exactly-zero** outputs, and digital silence *legitimately* produces exactly-zero encoder output and a NaN detector score. | Sentinel T2.6 checks for zeros **on a non-silent fixture**. `detectWatermark` guards `maskSum == 0`. `applyWatermark(zeros)` is asserted to return exactly zeros (E3.3). Range bound is *derived*, not sampled: `norm ≤ 1.232`, max activation 1.38 (enc) / 13.5 (dec) — nowhere near 65504. | None; this is a diagnosis-discipline risk, handled by the sentinel. |
| **R8** | **Load-time / SIGBUS / cache regressions on device.** `MLModel.compileModel(at:)` returns a fresh tmp `.mlmodelc` each launch ⇒ the aned E5 bundle cache (keyed by *compiled path*) misses ⇒ full ANE AOT compile every launch. A failed load SIGBUSes uncatchably. Xcode's `PBXFileSystemSynchronizedRootGroup` **recurses into subdirectories**, so a stray model in `Models/_hold/` still ships and can crash the probe before the model under test runs. | Port `ChatterboxCoreMLModel.compiledModel(at:cacheDir:)` **verbatim** (path-independent identity key + FNV-1a package stamp + `coremldata.bin` completeness sentinel + `excludeFromBackup` + `pruneStaleSiblings`). One (model × CU) per probe process. Excluded models live in a **sibling** `_excluded_models/`, never a subdirectory. `find <ProbeApp> -name '*.mlpackage'` before interpreting any device result. | Ship precompiled `.mlmodelc` in the HF repo (the loader already globs both extensions). |

---

## 9. Known, accepted limitations (document, do not gate on)

1. **Band-limited / high-passed audio (no energy below ~2.5 kHz).** The 0–2 kHz subband sits at the fp32 STFT noise floor, so the *normalized-dB* subband diff between torch and vDSP explodes to **1.3e-1** while the audio-domain agreement still holds at **110 dB SNR**. The detector score can shift by **9.4e-3** on such input, and Python's own score there is ~0.55 — 0.05 from flipping the rounded bit. **Gate on the time domain, never on the subband magspec.** TTS output is not band-limited; this is out of distribution. Measure it, report it, do not gate on it.
2. **Detector robustness is thin under attack, in Python itself.** The reference detector already **misses** (score < 0.5) at 20 dB additive-noise SNR on 2/8 seeds, and at 16 dB on 8/8. The fp16/ANE path shifts the score by **+0.002**, ~25× below the reference's own seed-to-seed spread (±0.05). The claim "the detector's margin is ~0.5" is false; the correct statement is that **clean-audio detection is the only meaningful gate**, and there the margin is ~370×.
3. **Detector false-positives on non-speech.** An unwatermarked tone+noise clip scores **0.9507 → rounds to 1**. Real speech scores 8e-06. Never validate detector `round()` agreement on synthetic signals.
4. **24→32→24 round trip is capped at ~44.2 dB SNR.** Inherent (the trip band-limits at ~11.2 kHz); Python has it too (44.23 dB). Not our bug.
5. **`cos ≥ 0.9999` is a real gate; "Python still detects it" is not.** Measured: the detector still returns 1.0000 with a **100 % relative residual error**. `cos` / `ΔSNR` are the discriminating gates; the detector check is a smoke test.
6. **`generateStream` chunks are unwatermarked** under the whole-utterance hook. Flag to the user.

---

## 10. Repo layout

```
perth-coreml/
├── Package.swift                         # swift-tools 6.0; .macOS(.v15), .iOS(.v18); dep: swift-transformers from "1.3.0"
├── README.md
├── CLAUDE.md                             # working style + the traps in §5 + the device-log discipline
├── docs/
│   ├── porting-plan.md                   # this document
│   ├── model-contract.md                 # §3.3, frozen
│   ├── parity-gates.md                   # §1.2/§1.3 tables + how to run them
│   └── device-log.md                     # append-only: every device measurement, with what was ACTUALLY bundled
├── Sources/
│   ├── PerthCoreML/
│   │   ├── PerthWatermarker.swift        # public actor (§7)
│   │   ├── PerthEncoderRunner.swift      # MLModel wrapper, non-stateful MLDictionaryFeatureProvider path
│   │   ├── PerthDecoderRunner.swift      # 3 MLModel instances (functionName: slow/normal/fast)
│   │   ├── TileScheduler.swift           # §3.1 tiler; shared by encoder + all 3 decoder branches
│   │   ├── AudioProcessor.swift          # dB normalize/denormalize, magmask, residual add
│   │   ├── STFT.swift                    # vDSP fft_zrip; reflect pad; true OLA envelope (§5.1/§5.3)
│   │   ├── Interpolate.swift             # torch-exact linear (fp32 scale + fmaf) + nearest (§5.4)
│   │   ├── PolyphaseResampler.swift      # vDSP_desamp engine (§5.7)
│   │   ├── PerthResamplerTaps.swift      # hq801 (apply) + scipy81 (detect), generated
│   │   ├── HannWindow.swift              # loads Resources/hann2048.f32
│   │   ├── ModelRepository.swift         # HF Hub download + dual-extension discovery
│   │   ├── CompiledModelCache.swift      # ported verbatim from ChatterboxCoreMLModel.compiledModel(at:cacheDir:)
│   │   ├── MLMultiArray+Helpers.swift    # float32/float16 builders — #if arch(arm64) guard on ALL Float16 touches
│   │   ├── Constants.swift  Errors.swift  Log.swift
│   │   └── Resources/hann2048.f32        # 8192 bytes, exported from torch.hann_window(2048)
│   └── PerthCLI/main.swift               # watermark | detect | dump-intermediates (drives the parity harness)
├── Tests/PerthCoreMLTests/               # swift-testing (@Test/#expect/@Suite) — zero XCTest
│   ├── STFTTests.swift  ISTFTTests.swift  SilenceTests.swift
│   ├── InterpolateTests.swift  MagmaskTests.swift  SoftmaxRoundTests.swift
│   ├── TileSchedulerTests.swift           # whole vs tiled -> byte-identical
│   ├── ResamplerTests.swift               # incl. n%3==1 (the fix_length zero) and up=3 tap rounding
│   ├── CompiledModelCacheTests.swift      # fakePackage(in:name:weightBytes:) trick
│   ├── ParityTests.swift                  # gated on PERTH_MODEL_DIR; Tier-1 + Tier-2 tables
│   └── Fixtures/                          # .npy oracles; read via #filePath (matches `exclude:` in Package.swift)
├── converter/                             # pinned: torch==2.8.0, coremltools==9.0, librosa==0.11.0, soxr==1.1.0, numpy==2.4.6
│   ├── perth_ref.py  make_corpus.py  dump_fixtures.py
│   ├── export_encoder.py  export_decoder.py
│   ├── validate_mac.py                    # Tier-1/Tier-2 gates via coremltools predict, all 4 CUs
│   ├── generate_taps.py  extract_soxr_kernel.py    # CI: re-derive 801 taps from pinned soxr, assert <=1e-5
│   ├── interp_fma_sweep.py                # CI: re-run the 3000-case FMA sweep on any torch upgrade
│   └── pocketfft_probe/                   # E0.4, time-boxed; delete if it doesn't byte-match
├── repro/
│   ├── Package.swift                      # macOS executableTarget "PerthProbe" (Mac CU sweep, Tier-1 pre-gate)
│   ├── Sources/PerthProbe/main.swift      # data-driven from probe_meta.json; hand-rolled ~30-line NPY reader
│   ├── PerthProbe-iOS/PerthProbe.xcodeproj  # minimal SwiftUI app, one PerthProbeApp.swift, os.Logger only
│   ├── _excluded_models/                  # SIBLING of the synced root, never a subdirectory
│   └── run_probe.sh                       # idevicesyslog started BEFORE launch; one model x CU per process
└── fixtures/corpus/                       # ~30 wavs + manifest.json (§2)
```

`Package.swift` uses `exclude: ["Fixtures"]` on the test target (fixtures are read by absolute path via `#filePath`, matching chatterbox's deliberate choice) — **not** `resources: [.copy(...)]`. Pick one; do not write `exclude:` and then call `Bundle.module`.

---

## 11. Where the critiques disagreed with the reports — rulings

| # | Dispute | Ruling | Why |
|---|---|---|---|
| **1** | **Tiling.** `coreml` dive: vmask-after-every-layer, `W=1024`, handles `T<W`. `decoder` dive + `integration` critique: edge-anchored, no-explicit-pad, requires `T ≥ W`. | **vmask wins.** | Both are bit-exact (`torch.equal == True` / `max_abs_err = 0.0`). But edge-anchored's binding constraint is `floor(3T/4) ≥ W` (the `integration` report and the `decoder` report both got this wrong; only `decoder`'s critique D3 caught it) ⇒ at `W=128` you need 1.70 s of audio, and `T_fast` bottoms out at 3 ⇒ **no fixed `W` covers all `T`** ⇒ a second static shape or a CPU fallback. vmask needs neither, keeps a single static shape, and the `mul` stays on ANE. Strictly dominant. |
| **2** | **The lead's original zero-pad sketch.** | **Dead.** Every reviewer independently killed it. | `Conv(128,256,k=1)` has a bias ⇒ zero input frames become `LeakyReLU(b₀)`, absmax **0.0225** vs a watermark residual of **0.017**. The artifact is *larger than the watermark*, corrupts exactly frames `T−15..T−1`, and is **invariant to halo size**. |
| **3** | **fp16 safety.** `coreml`/`decoder` dives: "range is safe ⇒ the split is right." `decoder` critique D1 / `integration` critique M1: "you checked range, not precision; Mac ANE burns 85 % of the SNR budget." | **The critiques win on framing; the dives win on range.** | Range genuinely is safe (max activation 1.38 enc / 13.5 dec vs 65504; the bound is *derivable*: `norm ≤ 1.232`). But that was never the risk. The risk is **fp16 accumulation across 1792 MACs producing a 0.017-magnitude output from O(1) intermediates** — catastrophic cancellation. CoreML-fp16 is 35× worse than torch's `.half()`. The three reported Mac-ANE `ΔSNR` values (−0.019 / −0.036 / −0.085 dB) disagree 4× ⇒ **E1.4 is the tiebreak, with a 0.05 dB stop-loss.** |
| **4** | **Detect-path resampler.** `resampler` dive: one `hq` filter everywhere. Critique S4: use `scipy81` on detect to match Python's `res_type="polyphase"`. | **S4 wins.** | The gate is literally "Swift detector agrees with Python detector on the same input". Swapping the detect resampler shifts the `magmask` threshold (43.6 → 38.3) and **flips mask bits** (1/651 measured). `hq` on detect buys nothing (the detector saturates) and adds an unbounded deviation to a `round()`-thresholded output. Ship both prototypes; match Python on both legs. |
| **5** | **"Bins 0–127 are all that matter, so the resampler swap is structurally harmless."** | **Critique S2 wins: the mechanism is false, the conclusion holds.** | `magmask` sums **all 1025 bins** in the *dB* domain, which amplifies stopband differences enormously (mean normalized-dB in bins 750–1024: 0.177 soxr vs 0.485 polyphase). Never reason from "the watermark lives in 0–2 kHz" again. |
| **6** | **"Detection saturates at raw = 1.000000."** | **Critique S3 wins.** | That "raw" is *post*-`clip(0,1)`. The true pre-clip score is 1.000916. Every comparison in the resampler dive's "decisive experiment" was pinned to a ceiling and carried **zero information**. **All detector comparisons in this plan use the pre-clip scalar.** |
| **7** | **"319 trailing samples never enter the STFT."** | **Critique D6 wins: false.** | `N − 320·(N//320) ≤ 319 < 1024`, so every original sample is inside a frame for all `N ≥ 1025` (exhaustively checked). Only the ISTFT *output* is truncated. The claim invites someone to "fix" a non-existent information loss. |
| **8** | **"Byte match remains achievable for the deterministic scalar ops (dB, normalize, magmask, softmax)."** | **Critique D4 wins.** | The same Sleef-vs-libm argument that kills the `cosf` window applies verbatim to `log10` (99.60 % bitwise), `10**x` (98.04 %), `atan2` (94.31 %). Only `normalize`, `magmask`, and FMA-free arithmetic are bitwise. |
| **9** | **"Multifunction ⇒ one file, one load."** | **Critique D7 wins.** | `MLModelConfiguration.functionName` binds one `MLModel` to one function ⇒ **three loads, three AOT compiles** — chatterbox's own `CLAUDE.md:246` says so. And there is **zero weight dedup** (1,125 bytes saved out of 18.7 MB). Multifunction survives only as file tidiness + a single cache key. |
| **10** | **"The detector's margin is ~0.5."** | **Critique D6 (coreml) wins.** | Under mild additive noise the reference detector's own operating margin is **~0.04**, and it already misses at ≤20 dB SNR. The correct fp16 justification is the *measured* one: the fp16/ANE path shifts the score by **+0.002**, 25× below the reference's own ±0.05 seed spread. |
| **11** | **"Per-chunk watermarking is also exact (stateless convs)."** | **Critique H2 wins: false.** | Measured `cos(whole, per-chunk) = 0.999975`, `max|Δ| = 0.0103` vs a watermark amplitude of 0.0133 — **77 % seam error**. Statelessness is irrelevant; the breakage is reflect-padding, ISTFT truncation, and `magmask`'s global max. **Whole-utterance only.** |
| **12** | **"print()+fflush; os.Logger never reaches `--console`."** (from the brief) | **`integration` dive + critique win: stale.** | `idevicesyslog` streams `os.Logger` `.notice`+`.debug` on iOS 26.5 (chatterbox `CLAUDE.md`, corrected 2026-05-29); every current probe uses `os.Logger` exclusively. The real constraint is that `devicectl --console` only relays stdout. Build `PerthProbe` on `os.Logger` + `idevicesyslog -u <UDID> --process PerthProbe`. |
| **13** | **ANE cost/benefit.** `coreml` dive §7: "I could not measure cold compile without the phone." Critique D10: "you could, and it already fails your own threshold." | **D10 wins.** | Mac, fresh process, warm cache: **75 ms (cpuOnly) / 158 ms (.all)**. ANE costs **+84 ms of load to save 0.85 ms per 10 s window** — break-even at ~100 utterances/process. **ANE is a stated success criterion, not a performance need.** Say so out loud; the default-CU decision (E5.2) is a data call, and CPU-fp32 may well be the right production default with ANE shipped as the proven artifact. |
| **14** | **Rescaling / recentering as an fp16 mitigation.** | **`decoder` critique D1 wins: dead on arrival.** | Float is scale-invariant and input quantization contributes only 0.002 %. Folding a gain into the k=1 convs cannot fix cancellation. Do not spend a day on it. Go to the hybrid split (R1-b) instead. |
| **15** | **Decoder guard: `T ≥ 2` vs `T ≥ 3`.** | **Both are moot; implement the `D == 1 ⇒ scale = 0` interp case anyway.** | `N32 ≥ 1025` (a hard STFT constraint) ⇒ `T ≥ 4` ⇒ `T_fast ≥ 3` ⇒ neither degenerate case is reachable via the audio path. But `D == 1` is what ATen's `area_pixel_compute_scale` handles, and the conv stack may be called directly in tests. |