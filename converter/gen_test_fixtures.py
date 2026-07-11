"""Emit the golden vectors the Swift test suite asserts against.

Every fixture here is produced by the *same* library call the Swift code claims to reproduce --
torch's `F.interpolate`, torchaudio's `Spectrogram`/`InverseSpectrogram`, scipy's `resample_poly`,
librosa's default (soxr_hq) `resample`. A Swift test that only checks Swift against Swift would
have passed on every bug this port actually hit.

The STFT window is read back out of `PerthAssets.swift` rather than out of the checkpoint, so this
script needs no Perth install -- and so the fixtures also pin "the window we ship is the window
torch was fed".

Output: Tests/PerthCoreMLTests/Fixtures/*.json, with float arrays as base64 little-endian blobs
(the same encoding `PerthAssets` already uses, so Swift needs no dependency to read them).

Pinned: torch 2.8.0. The linear-interpolation golden is bit-exact against *that* build's kernel;
re-run this and the Swift suite on any torch upgrade.
"""
from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torchaudio.transforms import InverseSpectrogram, Spectrogram

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "Sources/PerthCoreML/PerthAssets.swift"
DST = ROOT / "Tests/PerthCoreMLTests/Fixtures"

N_FFT, HOP, SUBBAND, HALO = 2048, 320, 128, 15
STFT_MAGNITUDE_MIN = 1e-9
MIN_LEVEL_DB = 20.0 * np.log10(STFT_MAGNITUDE_MIN)      # -180.0
DENORM_SCALE = -MIN_LEVEL_DB + 15.0                     # 195.0


def b32(a) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


def i32(a) -> str:
    return base64.b64encode(np.asarray(a, dtype="<i4").tobytes()).decode()


def checkpoint_window() -> torch.Tensor:
    m = re.search(r'stftWindow: \[Float\] = decodeF32\("([^"]+)"\)', ASSETS.read_text())
    w = np.frombuffer(base64.b64decode(m.group(1)), dtype="<f4")
    assert w.shape == (N_FFT,), w.shape
    return torch.from_numpy(w.copy())


def white(n: int, seed: int) -> np.ndarray:
    """Full-band noise, for the STFT golden.

    Band-limited noise would leave the top bins at the 1e-9 magnitude floor, where `atan2` is
    reading numerical dust -- the phase comparison would then measure pocketfft's round-off
    against vDSP's rather than anything about the port.
    """
    rng = np.random.default_rng(seed)
    return (0.5 * rng.standard_normal(n)).astype(np.float32)


def bandlimited(n: int, sr: int, seed: int, hz: float = 8000.0) -> np.ndarray:
    """Noise with no energy above `hz`.

    White noise would make the soxr-vs-Kaiser comparison a test of the two filters' transition
    bands rather than of the Swift engine. Real speech is naturally lowpass, which is why the
    project measured >100 dB SNR on it in the first place.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec[freqs > hz] = 0
    y = np.fft.irfft(spec, n=n)
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


# -- interpolation ---------------------------------------------------------------------------

def interp_pairs() -> list[tuple[int, int]]:
    """(T, S) pairs, with the detect path's own branch lengths well represented."""
    pairs = set()
    for t in (2, 3, 5, 7, 22, 30, 63, 64, 65, 100, 101, 127, 254, 301, 376, 501, 999, 1000, 1024):
        for s in (5 * t // 4, t, 3 * t // 4):       # exactly Python's int(t*1.25) / int(t*0.75)
            if s >= 1:
                pairs.add((t, s))
    # The pairs the index math is known to disagree on between fp32 and fp64.
    pairs |= {(30, 22), (301, 376), (301, 225), (254, 190), (95, 254), (13, 11), (17, 23)}
    return sorted(pairs)


def gen_interp() -> dict:
    rng = np.random.default_rng(7)
    linear, nearest = [], []
    for t, s in interp_pairs():
        for c in ((1, 4) if t <= 101 else (1,)):
            x = rng.standard_normal((1, c, t)).astype(np.float32)
            y = F.interpolate(torch.from_numpy(x), size=s, mode="linear", align_corners=True)
            linear.append({"channels": c, "tIn": t, "tOut": s,
                           "x": b32(x.reshape(-1)), "y": b32(y.numpy().reshape(-1))})

        x = rng.standard_normal((1, 1, t)).astype(np.float32)
        y = F.interpolate(torch.from_numpy(x), size=s, mode="nearest")
        # A ramp input makes torch's chosen source index directly observable: out[d] == idx[d].
        ramp = torch.arange(t, dtype=torch.float32).reshape(1, 1, t)
        idx = F.interpolate(ramp, size=s, mode="nearest").numpy().reshape(-1)
        nearest.append({"tIn": t, "tOut": s, "x": b32(x.reshape(-1)),
                        "y": b32(y.numpy().reshape(-1)), "idx": i32(idx)})

    # The specific case where fp64 index math lands on a different source sample than fp32.
    probe = F.interpolate(torch.arange(30, dtype=torch.float32).reshape(1, 1, 30),
                          size=22, mode="nearest").numpy().reshape(-1)
    assert probe[11] == 15.0, probe[11]
    return {"torch": torch.__version__, "linear": linear, "nearest": nearest,
            "fp32IndexProbe": {"tIn": 30, "tOut": 22, "d": 11, "sourceIndex": int(probe[11])}}


# -- resampler -------------------------------------------------------------------------------

def gen_resample() -> dict:
    poly, hq_up, hq_down = [], [], []

    # n % 3 == 1 is the length where librosa's fix_length appends a zero (soxr's own output is
    # round(n*4/3), which falls one short of the ceil librosa demands). n % 3 == 2 rounds UP and
    # gets no zero -- the two cases must be told apart.
    for n in (1024, 6000, 6001, 6002, 12005):
        x = bandlimited(n, 24000, seed=n)
        poly.append({"n": n, "x": b32(x),
                     "y": b32(resample_poly(x, 4, 3).astype(np.float32))})

    for n in (1024, 6000, 6001, 6002, 24001):
        x = bandlimited(n, 24000, seed=n + 1)
        y = librosa.resample(x, orig_sr=24000, target_sr=32000)      # default res_type: soxr_hq
        assert len(y) == math.ceil(n * 4 / 3)
        hq_up.append({"n": n, "x": b32(x), "y": b32(y),
                      "pythonLastIsZero": bool(y[-1] == 0.0)})

    # `down32to24Apply` is only ever handed an ISTFT output, which is a whole number of hops.
    for n in (1280, 6400, 12800):
        assert n % HOP == 0
        x = bandlimited(n, 32000, seed=n + 2)
        y = librosa.resample(x, orig_sr=32000, target_sr=24000)
        hq_down.append({"n": n, "x": b32(x), "y": b32(y)})

    return {"polyUp24to32": poly, "hqUp24to32": hq_up, "hqDown32to24": hq_down}


# -- tiling ----------------------------------------------------------------------------------

def tile_plan(t: int, w: int, halo: int = HALO):
    stride = w - 2 * halo
    out_start = 0
    while out_start < t:
        in_start = 0 if out_start == 0 else out_start - halo
        keep_lo = out_start - in_start
        n_real = min(w, t - in_start)
        n_out = min(stride if out_start else w - halo, t - out_start, n_real - keep_lo)
        yield in_start, keep_lo, n_out, n_real
        out_start += n_out


def gen_tiling() -> dict:
    cases = []
    for w in (1024, 256):
        ts = {1, 2, HALO, w - 1, w, w + 1, w + 2, 2 * w, 2 * w + 1, 3 * w, 5 * w // 2,
              w - HALO, w - HALO + 1, w - 2 * HALO, w - 2 * HALO + 1}
        ts |= {int(t * f) for t in (301, 1001, 3001) for f in (1.25, 1.0, 0.75)}
        for t in sorted(x for x in ts if x >= 1):
            cases.append({"w": w, "halo": HALO, "t": t,
                          "tiles": [list(map(int, x)) for x in tile_plan(t, w)]})
    return {"cases": cases}


# -- STFT ------------------------------------------------------------------------------------

def gen_stft(window: torch.Tensor) -> dict:
    wf = lambda _n, **_kw: window.clone()
    spec = Spectrogram(n_fft=N_FFT, win_length=N_FFT, power=None, hop_length=HOP,
                       window_fn=wf, normalized=False)
    ispec = InverseSpectrogram(n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                               window_fn=wf, normalized=False)

    cases = []
    for name, sig in (("min", white(1025, seed=11)),
                      ("odd", white(3201, seed=12)),
                      ("noise", white(6400, seed=13)),
                      ("silence", np.zeros(6400, dtype=np.float32))):
        cx = spec(torch.from_numpy(sig))
        mag = 20 * torch.log10(cx.abs().clip(STFT_MAGNITUDE_MIN))
        mag = (mag - MIN_LEVEL_DB) / DENORM_SCALE
        phase = torch.angle(cx)
        lin = 10.0 ** ((mag * DENORM_SCALE + MIN_LEVEL_DB) / 20).clip(max=10)
        back = ispec(lin * torch.exp(1.0j * phase))

        t = cx.shape[1]
        assert t == len(sig) // HOP + 1 and len(back) == HOP * (t - 1)
        cases.append({"name": name, "n": len(sig), "frames": t,
                      "signal": b32(sig), "mag": b32(mag.numpy().reshape(-1)),
                      "phase": b32(phase.numpy().reshape(-1)), "istft": b32(back.numpy())})
    return {"nFFT": N_FFT, "hop": HOP, "nFreq": N_FFT // 2 + 1, "cases": cases}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    window = checkpoint_window()
    for name, data in (("interp", gen_interp()),
                       ("resample", gen_resample()),
                       ("tiling", gen_tiling()),
                       ("stft", gen_stft(window))):
        p = DST / f"{name}.json"
        p.write_text(json.dumps(data))
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
