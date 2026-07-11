"""Where does each op actually run? Asks CoreML for the compute plan.

Mac ANE placement is only a smoke signal -- iPhone ANE differs, and the real gate is the
on-device probe. But a model that CoreML refuses to put on the ANE *here* will not
magically land there on device, so this is the cheap disconfirming test.
"""
import argparse, sys
from collections import Counter
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def plan_for(pkg: Path):
    """MLComputePlan wants a compiled .mlmodelc, not the .mlpackage."""
    import subprocess, tempfile
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(tmp)],
                   check=True, capture_output=True)
    mlmodelc = next(tmp.glob("*.mlmodelc"))
    return MLComputePlan.load_from_path(path=str(mlmodelc), compute_units=ct.ComputeUnit.ALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../out")
    ap.add_argument("--models", nargs="+",
                    default=["PerthEncoder.mlpackage", "PerthDecoder.mlpackage"])
    args = ap.parse_args()

    for name in args.models:
        pkg = Path(args.out) / name
        try:
            plan = plan_for(pkg)
        except Exception as e:
            print(f"{name}: compute plan unavailable ({type(e).__name__}: {e})")
            continue

        prog = plan.model_structure.program
        fn = prog.functions["main"]
        short = lambda d: type(d).__name__.replace("ML", "").replace("ComputeDevice", "")
        preferred, supported, by_op = Counter(), Counter(), Counter()
        for op in fn.block.operations:
            if op.operator_name == "const":
                continue
            du = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if du is None:
                preferred["unplaced"] += 1
                by_op[f"{op.operator_name} -> unplaced"] += 1
                continue
            p = short(du.preferred_compute_device)
            sup = "+".join(sorted(short(d) for d in du.supported_compute_devices))
            preferred[p] += 1
            supported[sup] += 1
            by_op[f"{op.operator_name} -> {p}  (can: {sup})"] += 1

        n = sum(preferred.values())
        print(f"\n{name}  ({n} compute ops, consts excluded)")
        print("  preferred: " + ", ".join(f"{c} {d}" for d, c in preferred.most_common()))
        print("  supported: " + ", ".join(f"{c}x [{s}]" for s, c in supported.most_common()))
        for k, c in by_op.most_common():
            print(f"     {c:4d}  {k}")


if __name__ == "__main__":
    sys.exit(main())
