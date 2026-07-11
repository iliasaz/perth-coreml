"""Gate 1: the tiled/windowed decomposition must reproduce stock Perth bit-for-bit.

This isolates "did we decompose the model correctly?" from "did CoreML convert it
faithfully?". No CoreML here -- the conv stacks still run in PyTorch.

Targets:
  apply()  -- max|err| == 0.0 exactly (the watermarked waveform is the shipped artefact)
  detect() -- |delta| < 1e-5 and the same rounded decision. Exact equality is not expected:
              we accumulate the masked-mean in float64 where stock uses a blocked fp32
              reduction, so the last couple of bits differ (in our favour).
"""
import sys, argparse
import numpy as np
import torch

sys.path.insert(0, "/Users/ilia/Developer/Perth/src")
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

from perth_pipeline import PerthPipeline, TorchRunner

CONF_TOL = 1e-5


def make_signal(n, sr=32000, seed=0, silence_s=0.2):
    """Speech-ish: tones + noise, with a leading silence so magmask has something to gate."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    s = (0.30 * np.sin(2 * np.pi * 220 * t) * np.exp(-((t - 1.0) ** 2) / 0.5)
         + 0.15 * np.sin(2 * np.pi * 700 * t) * (t > 0.5)
         + 0.08 * np.sin(2 * np.pi * 1500 * t) * (t > 1.2)
         + 0.02 * rng.standard_normal(n)).astype(np.float32)
    s[: min(int(silence_s * sr), n // 4)] = 0.0     # never silence the whole clip
    return s


def agree(a, b, tol):
    if np.isnan(a) and np.isnan(b):
        return True                                  # both NaN (silent audio) counts as agreement
    return abs(a - b) < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[256, 512, 1024])
    args = ap.parse_args()

    ref = PerthImplicitWatermarker(device="cpu")
    net = ref.perth_net
    window = net.ap.spectrogram.window.detach().clone()   # the CHECKPOINT's window, not a fresh hann

    cases = [4096, 16000, 32000, 96137, 160000, 320007]
    fails = 0

    for W in args.windows:
        print(f"\n{'=' * 78}\nWINDOW W={W}  (halo=15, stride={W - 30})")
        pipe = PerthPipeline(TorchRunner(net, W), window)
        for n in cases:
            sig = make_signal(n)

            gold_wav = ref.apply_watermark(sig, sample_rate=32000)
            got_wav = pipe.apply(sig)
            if gold_wav.shape != got_wav.shape:
                print(f"  n={n:7d}  SHAPE MISMATCH {gold_wav.shape} vs {got_wav.shape}")
                fails += 1
                continue

            gold_c = float(ref.get_watermark(gold_wav, sample_rate=32000, round=False))
            got_c = pipe.detect(got_wav, do_round=False)
            gold_r = float(ref.get_watermark(gold_wav, sample_rate=32000))
            got_r = pipe.detect(got_wav)

            e_wav = np.abs(gold_wav - got_wav).max()
            ok_wav = e_wav == 0.0
            ok_conf = agree(gold_c, got_c, CONF_TOL)
            ok_round = agree(gold_r, got_r, 1e-9)
            ok = ok_wav and ok_conf and ok_round
            fails += not ok
            t = n // 320 + 1
            print(f"  n={n:7d} T={t:5d} tiles={len(pipe._windows(t)):2d} "
                  f"| apply max|err|={e_wav:.3e} {'EXACT' if ok_wav else 'DRIFT'} "
                  f"| detect {gold_c:.7f} vs {got_c:.7f} (d={abs(gold_c - got_c):.1e}) "
                  f"| round {gold_r} vs {got_r} "
                  f"{'ok' if ok else '*** FAIL ***'}")

    print(f"\n{'=' * 78}")
    print("GATE 1:", "PASS -- decomposition reproduces stock Perth" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
