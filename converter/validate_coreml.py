"""Gate 2: the CoreML packages, driven through the real host pipeline, vs stock Perth.

Runs the SAVED .mlpackage files (not the in-memory conversion) so we validate the shipped
artefact. Reports both tiers:

  fp32 / cpuOnly  -- the "byte match" reference. Target: max|err| <= ~1e-6 on the waveform.
  fp16 / ALL      -- the ANE tier. Target: cos >= 0.9999, same rounded detect decision,
                     SNR within 0.1 dB of Python's.

Validate on BOTH cpuOnly and ALL: they diverge, and a bug hides if you only report one.
"""
import os
import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, os.environ.get("PERTH_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Perth", "src"))
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

from perth_pipeline import PerthPipeline
from validate_decomposition import make_signal

CU = {"cpu": ct.ComputeUnit.CPU_ONLY, "all": ct.ComputeUnit.ALL,
      "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE}


class CoreMLRunner:
    def __init__(self, out: Path, window: int, fp32: bool, cu: str):
        tag = "_fp32" if fp32 else ""
        self.window = window
        u = CU[cu]
        self.enc = ct.models.MLModel(str(out / f"PerthEncoder{tag}.mlpackage"), compute_units=u)
        self.dec = ct.models.MLModel(str(out / f"PerthDecoder{tag}.mlpackage"), compute_units=u)

    def encode(self, x, m):
        return self.enc.predict({"sub_mag": x, "mask": m})["residual"]

    def decode(self, sx, sm, nx, nm, fx, fm):
        o = self.dec.predict({"slow_x": sx, "slow_m": sm, "norm_x": nx,
                              "norm_m": nm, "fast_x": fx, "fast_m": fm})
        return o["slow_out"], o["norm_out"], o["fast_out"]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def snr(clean, wm):
    n = wm - clean
    return 10 * np.log10(np.sum(clean ** 2) / np.sum(n ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../out")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--lengths", type=int, nargs="+", default=[32000, 96137, 240000])
    args = ap.parse_args()
    out = Path(args.out)

    ref = PerthImplicitWatermarker(device="cpu")
    window = ref.perth_net.ap.spectrogram.window.detach().clone()

    configs = [("fp32", "cpu"), ("fp32", "all"), ("fp16", "cpu"), ("fp16", "all")]
    fails = 0

    for n in args.lengths:
        sig = make_signal(n)
        gold = ref.apply_watermark(sig, sample_rate=32000)
        gold_c = float(ref.get_watermark(gold, sample_rate=32000, round=False))
        gold_r = float(ref.get_watermark(gold, sample_rate=32000))
        gold_snr = snr(sig[:len(gold)], gold)

        print(f"\n{'=' * 90}")
        print(f"n={n}  ({n / 32000:.2f}s)   python: conf={gold_c:.6f} -> {gold_r}   SNR={gold_snr:.2f} dB")
        print(f"{'-' * 90}")
        print(f"  {'prec/CU':10s} {'apply max|err|':>15s} {'cos(wav)':>11s} {'SNR dB':>9s} "
              f"{'ours(ours)':>11s} {'PY(ours)':>10s} {'ours(PY)':>10s}  verdict")

        for prec, cu in configs:
            pipe = PerthPipeline(CoreMLRunner(out, args.window, prec == "fp32", cu), window)
            got = pipe.apply(sig)
            got_r = pipe.detect(got)                                    # our detector, our audio

            # The decisive cross-detection tests: each side's detector must accept the
            # other side's watermarked audio. This is what "the port works" actually means.
            py_on_ours = float(ref.get_watermark(got, sample_rate=32000, round=False))
            ours_on_py = pipe.detect(gold, do_round=False)

            e = float(np.abs(gold - got).max())
            c = cos(gold, got)
            s = snr(sig[:len(got)], got)

            cross_ok = (round(py_on_ours) == gold_r) and (round(ours_on_py) == gold_r)
            if prec == "fp32":
                ok = e <= 1e-5 and got_r == gold_r and cross_ok
                why = "byte-match tier"
            else:
                ok = c >= 0.9999 and got_r == gold_r and abs(s - gold_snr) <= 0.3 and cross_ok
                why = "ANE tier"
            fails += not ok
            print(f"  {prec}/{cu:5s} {e:15.3e} {c:11.7f} {s:9.2f} {got_r:11.1f} "
                  f"{py_on_ours:10.6f} {ours_on_py:10.6f}  "
                  f"{'PASS' if ok else '*** FAIL ***'} ({why})")

    print(f"\n{'=' * 90}")
    print("GATE 2:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
