"""Decisive experiment for the resampler decision.

Python's apply_watermark resamples 24k->32k with librosa's DEFAULT res_type (soxr_hq) and
back; get_watermark uses res_type="polyphase". We are not vendoring libsoxr, so Swift will
use a scipy-resample_poly-equivalent polyphase FIR in BOTH directions.

The question this answers: does that substitution break anything? Specifically --
  1. how far apart are soxr_hq and resample_poly on the same signal?
  2. if we watermark a 24 kHz clip using polyphase resampling, does STOCK Perth (which
     resamples with polyphase on the detect path) still detect it?
  3. does the round trip preserve length?

Also dumps the exact FIR taps scipy would design, so Swift needs no filter-design code.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import firwin, resample_poly, upfirdn

sys.path.insert(0, "/Users/ilia/Developer/Perth/src")
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

SR_IN, SR_PERTH = 24000, 32000


def design(up, down):
    """Exactly what scipy.signal.resample_poly designs internally (window=('kaiser', 5.0))."""
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    return up * h, half_len


def resample_poly_manual(x, up, down):
    """Re-derivation of scipy's resample_poly, to prove Swift can reproduce it from taps alone."""
    n_in = len(x)
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    h, half_len = design(up, down)
    n_pre_pad = down - half_len % down
    n_pre_remove = (half_len + n_pre_pad) // down
    n_post_pad = 0

    def out_len(len_h):
        return (((n_in - 1) * up + len_h) - 1) // down + 1

    while out_len(len(h) + n_pre_pad + n_post_pad) < n_out + n_pre_remove:
        n_post_pad += 1
    hh = np.concatenate((np.zeros(n_pre_pad), h, np.zeros(n_post_pad)))
    y = upfirdn(hh, x, up, down)
    return y[n_pre_remove: n_pre_remove + n_out], dict(
        up=up, down=down, taps=[float(v) for v in hh],
        n_pre_remove=int(n_pre_remove), n_pre_pad=int(n_pre_pad), n_post_pad=int(n_post_pad))


def main():
    wav_path = "/Users/ilia/Developer/Chatterbox-TTS-Server/voices/Abigail.wav"
    y, sr = librosa.load(wav_path, sr=SR_IN, mono=True)
    y = y[: SR_IN * 6].astype(np.float32)
    print(f"real speech: {wav_path.split('/')[-1]}  {len(y)} samples @ {SR_IN} Hz "
          f"({len(y)/SR_IN:.2f}s)\n")

    # --- 1. how different are the two resamplers?
    up32 = librosa.resample(y, orig_sr=SR_IN, target_sr=SR_PERTH)                   # soxr_hq
    up32_poly = librosa.resample(y, orig_sr=SR_IN, target_sr=SR_PERTH, res_type="polyphase")
    up32_mine, meta_up = resample_poly_manual(y.astype(np.float64), 4, 3)

    print("24k -> 32k:")
    print(f"  soxr_hq          len={len(up32)}")
    print(f"  polyphase        len={len(up32_poly)}   vs soxr: "
          f"cos={np.dot(up32, up32_poly)/np.linalg.norm(up32)/np.linalg.norm(up32_poly):.6f}  "
          f"max|d|={np.abs(up32-up32_poly).max():.2e}")
    print(f"  my resample_poly len={len(up32_mine)}   vs librosa polyphase: "
          f"max|d|={np.abs(up32_poly - up32_mine).max():.3e}   "
          f"{'BIT-MATCH' if np.abs(up32_poly-up32_mine).max() < 1e-6 else 'DIFFERS'}")

    # --- 2. round trip length + error
    back = resample_poly_manual(up32_mine, 3, 4)[0]
    print(f"\nround trip 24k->32k->24k: {len(y)} -> {len(up32_mine)} -> {len(back)}")

    # --- 3. THE DECISIVE TEST: watermark a 24k clip using ONLY polyphase resampling,
    #        then ask STOCK Perth (unmodified) whether it detects the watermark.
    wm = PerthImplicitWatermarker(device="cpu")

    gold24 = wm.apply_watermark(y, sample_rate=SR_IN)                     # stock: soxr both ways

    # our variant: polyphase up, watermark at 32k, polyphase down
    x32 = resample_poly_manual(y.astype(np.float64), 4, 3)[0].astype(np.float32)
    wm32 = wm.apply_watermark(x32, sample_rate=SR_PERTH)                 # no resample inside
    ours24 = resample_poly_manual(wm32.astype(np.float64), 3, 4)[0].astype(np.float32)

    print(f"\nstock  apply@24k -> {len(gold24)} samples")
    print(f"ours   apply@24k -> {len(ours24)} samples (polyphase both ways)")

    n = min(len(gold24), len(ours24))
    c = np.dot(gold24[:n], ours24[:n]) / np.linalg.norm(gold24[:n]) / np.linalg.norm(ours24[:n])
    print(f"cos(stock24, ours24) = {c:.6f}")

    print("\nDETECTION (stock Perth's own detector, which resamples with polyphase):")
    for name, a in (("clean 24k", y), ("stock-watermarked 24k", gold24),
                    ("ours-watermarked 24k (polyphase)", ours24)):
        raw = float(wm.get_watermark(a, sample_rate=SR_IN, round=False))
        rnd = float(wm.get_watermark(a, sample_rate=SR_IN))
        print(f"  {name:36s} conf={raw:.6f} -> {rnd}")

    # --- dump taps for Swift
    _, meta_dn = resample_poly_manual(np.zeros(16, dtype=np.float64), 3, 4)
    out = Path("../out")
    out.mkdir(exist_ok=True)
    (out / "perth_resampler.json").write_text(json.dumps({
        "note": "scipy.signal.resample_poly's internal Kaiser(5.0) firwin taps, already "
                "scaled by `up` and zero-padded exactly as scipy does. Swift applies plain "
                "upfirdn with these and trims n_pre_remove.",
        "up_24k_to_32k": meta_up, "down_32k_to_24k": meta_dn,
    }))
    print(f"\nwrote ../out/perth_resampler.json  "
          f"(up: {len(meta_up['taps'])} taps, down: {len(meta_dn['taps'])} taps)")


if __name__ == "__main__":
    main()
