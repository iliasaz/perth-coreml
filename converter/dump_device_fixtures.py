"""Dump the ground truth the on-device probe compares against.

Everything is raw little-endian float32 so Swift can read it with no dependency. The expected
outputs come from STOCK Python Perth, not from our CoreML models -- otherwise the probe would
be grading its own homework.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, "/Users/ilia/Developer/Perth/src")
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

DST = Path(__file__).parent.parent / "repro/PerthProbe/PerthProbe/Fixtures"
VOICE = "/Users/ilia/Developer/Chatterbox-TTS-Server/voices/Abigail.wav"
SECONDS = 5


def main():
    DST.mkdir(parents=True, exist_ok=True)
    wm = PerthImplicitWatermarker(device="cpu")

    y32, _ = librosa.load(VOICE, sr=32000, mono=True)
    y32 = y32[: 32000 * SECONDS].astype(np.float32)
    gold32 = wm.apply_watermark(y32, sample_rate=32000).astype(np.float32)

    y24, _ = librosa.load(VOICE, sr=24000, mono=True)
    y24 = y24[: 24000 * SECONDS].astype(np.float32)
    gold24 = wm.apply_watermark(y24, sample_rate=24000).astype(np.float32)

    meta = {
        "sr32": {
            "n_in": len(y32), "n_out": len(gold32),
            "detect_watermarked": float(wm.get_watermark(gold32, sample_rate=32000, round=False)),
            "detect_clean": float(wm.get_watermark(y32, sample_rate=32000, round=False)),
        },
        "sr24": {
            "n_in": len(y24), "n_out": len(gold24),
            "detect_watermarked": float(wm.get_watermark(gold24, sample_rate=24000, round=False)),
            "detect_clean": float(wm.get_watermark(y24, sample_rate=24000, round=False)),
        },
        "source": Path(VOICE).name,
        "note": "expected_* come from stock Python Perth on CPU, not from CoreML",
    }

    for name, arr in (("input32.f32", y32), ("expected32.f32", gold32),
                      ("input24.f32", y24), ("expected24.f32", gold24)):
        (DST / name).write_bytes(arr.astype("<f4").tobytes())
        print(f"  {name:18s} {len(arr):7d} samples")
    (DST / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta.json          {json.dumps(meta['sr32'])}")
    print(f"                     {json.dumps(meta['sr24'])}")
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
