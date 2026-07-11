"""Gate 3: the Swift package vs stock Python Perth, end to end, on real speech.

Runs the real `perth-cli` binary on real wav files and compares its output to what
`perth.PerthImplicitWatermarker` produces for the same input.

Tier 1 (fp32 / cpuOnly): numeric parity. Bitwise equality is not achievable and is not the
    gate -- torch links Sleef and Apple links libm, so `cos`/`atan2`/`log10` differ in the last
    bits, and vDSP's FFT uses a different summation order than pocketfft. The gate is the fp32
    agreement floor.
Tier 2 (fp16 / ANE): functional equivalence + cross-detection.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/ilia/Developer/Perth/src")
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

ROOT = Path(__file__).parent.parent
CLI = ROOT / ".build/debug/perth-cli"
VOICES = Path("/Users/ilia/Developer/Chatterbox-TTS-Server/voices")


def cos(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def snr(ref, other):
    n = min(len(ref), len(other))
    ref, other = ref[:n], other[:n]
    e = np.sum((ref - other) ** 2)
    return float("inf") if e == 0 else 10 * np.log10(np.sum(ref ** 2) / e)


def run_cli(wav, out, models, fp32, cu, detect=False):
    cmd = [str(CLI), str(wav), "--models", str(models), "--cu", cu, "--python-parity"]
    if fp32:
        cmd.append("--fp32")
    if detect:
        cmd.append("--detect")
    else:
        cmd += ["--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"perth-cli failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "out"))
    ap.add_argument("--sr", type=int, default=32000, choices=[24000, 32000])
    ap.add_argument("--clips", type=int, default=6)
    args = ap.parse_args()

    tmp = ROOT / "scratch/parity"
    tmp.mkdir(parents=True, exist_ok=True)
    wm = PerthImplicitWatermarker(device="cpu")

    voices = sorted(VOICES.glob("*.wav"))[: args.clips]
    configs = [("fp32", "cpu"), ("fp16", "ane"), ("fp16", "all")]

    print(f"input sample rate: {args.sr} Hz   "
          f"({'native -- no resampler in the path' if args.sr == 32000 else 'the chatterbox path'})")
    fails = 0

    for v in voices:
        y, _ = librosa.load(v, sr=args.sr, mono=True)
        y = y[: args.sr * 5].astype(np.float32)
        src = tmp / f"{v.stem}_{args.sr}.wav"
        sf.write(src, y, args.sr, subtype="FLOAT")

        gold = wm.apply_watermark(y, sample_rate=args.sr)
        gold_conf = float(wm.get_watermark(gold, sample_rate=args.sr, round=False))
        clean_conf = float(wm.get_watermark(y, sample_rate=args.sr, round=False))

        print(f"\n{v.stem}  ({len(y)} samples)   python: apply->{len(gold)}  "
              f"detect(wm)={gold_conf:.6f}  detect(clean)={clean_conf:.6f}")
        print(f"  {'cfg':10s} {'len':>7s} {'max|err|':>11s} {'cos':>11s} {'SNR dB':>8s} "
              f"{'swift detect':>13s} {'PY(swift)':>10s}  verdict")

        for prec, cu in configs:
            out = tmp / f"{v.stem}_{prec}_{cu}.wav"
            run_cli(src, out, args.models, prec == "fp32", cu)
            got, _ = sf.read(out, dtype="float32")

            e = float(np.abs(gold[: len(got)] - got[: len(gold)]).max())
            c = cos(gold, got)
            s = snr(gold, got)
            swift_conf = float(run_cli(out, None, args.models, prec == "fp32", cu, detect=True)
                               .split("raw=")[1].split()[0])
            py_on_swift = float(wm.get_watermark(got, sample_rate=args.sr, round=False))

            if prec == "fp32":
                ok = len(got) == len(gold) and s >= 90 and round(py_on_swift) == round(gold_conf)
            else:
                ok = len(got) == len(gold) and c >= 0.9999 and round(py_on_swift) == round(gold_conf)
            fails += not ok
            print(f"  {prec}/{cu:5s} {len(got):7d} {e:11.3e} {c:11.7f} {s:8.1f} "
                  f"{swift_conf:13.6f} {py_on_swift:10.6f}  "
                  f"{'PASS' if ok else '*** FAIL ***'}")

    print(f"\n{'=' * 96}")
    print("GATE 3:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
