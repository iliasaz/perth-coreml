"""Convert Perth-Net Implicit's conv stacks to CoreML at a static window width.

Emits, into --out:
  PerthEncoder{,_fp32}.mlpackage   (sub_mag, mask)                    -> residual
  PerthDecoder{,_fp32}.mlpackage   (slow/norm/fast x + mask)          -> slow/norm/fast out
  perth_assets.json                window taps + hparams, for the Swift host

Only the convolutions go to CoreML. STFT/ISTFT, dB (de)normalisation, magmask,
interpolation, masked-mean, softmax and the residual add stay on the host in fp32 -- CoreML
has no complex-number support, and those ops are free next to the conv stacks anyway.

The fp32 packages are the "byte match" reference tier; the fp16 packages are the ANE tier.
"""

from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, os.environ.get("PERTH_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Perth", "src"))
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

from perth_wrappers import HALO, DecoderStacks, EncoderStack

ENC_IN = ["sub_mag", "mask"]
ENC_OUT = ["residual"]
DEC_IN = ["slow_x", "slow_m", "norm_x", "norm_m", "fast_x", "fast_m"]
DEC_OUT = ["slow_out", "norm_out", "fast_out"]


def _convert(module, sample, in_names, out_names, w, precision, target):
    module = module.eval()
    traced = torch.jit.trace(module, sample, strict=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32)
                for n, t in zip(in_names, sample)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in out_names],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=target,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=512, help="static frame width W")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--target", default="iOS18", choices=["iOS17", "iOS18"])
    args = ap.parse_args()

    w = args.window
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = {"iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18}[args.target]

    net = PerthImplicitWatermarker(device="cpu").perth_net
    enc = EncoderStack(net.encoder)
    dec = DecoderStacks(net.decoder)

    x = torch.randn(1, 128, w)
    m = torch.ones(1, 1, w)
    enc_sample = (x, m)
    dec_sample = (x, m, x.clone(), m.clone(), x.clone(), m.clone())

    for tag, precision in (("", ct.precision.FLOAT16), ("_fp32", ct.precision.FLOAT32)):
        for name, module, sample, ins, outs in (
            ("PerthEncoder", enc, enc_sample, ENC_IN, ENC_OUT),
            ("PerthDecoder", dec, dec_sample, DEC_IN, DEC_OUT),
        ):
            mlmodel = _convert(module, sample, ins, outs, w, precision, target)
            mlmodel.short_description = (
                f"Perth-Net Implicit {name.replace('Perth', '').lower()} conv stack, "
                f"static window W={w}, halo={HALO}. Activations are masked after every layer, "
                f"which makes a zero-padded window bit-identical to full-length PyTorch."
            )
            mlmodel.user_defined_metadata["perth.window"] = str(w)
            mlmodel.user_defined_metadata["perth.halo"] = str(HALO)
            p = out / f"{name}{tag}.mlpackage"
            mlmodel.save(str(p))
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
            print(f"  saved {p.name:28s} {size:7.1f} MB")

    window = net.ap.spectrogram.window.detach().cpu().numpy().astype(np.float64)
    assets = {
        "note": "window comes from the CHECKPOINT (load_state_dict overwrites torchaudio's "
                "fresh hann); it differs from torch.hann_window(2048) by 1 ULP and that "
                "difference is visible in the output waveform.",
        "sample_rate": 32000, "n_fft": 2048, "hop_size": 320, "window_size": 2048,
        "subband": 128, "n_freq": 1025, "stft_magnitude_min": 1e-9,
        "min_level_db": -180.0, "denorm_scale": 195.0, "magmask_p": 0.05,
        "window_frames": w, "halo": HALO,
        "stft_window": [float(v) for v in window],
    }
    (out / "perth_assets.json").write_text(json.dumps(assets))
    print(f"  saved perth_assets.json      ({len(window)} window taps)")


if __name__ == "__main__":
    main()
