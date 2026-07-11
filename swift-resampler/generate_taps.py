import numpy as np
from scipy.signal import firwin
HQ  = firwin(801, 11482/48000.0, window=('kaiser',13.75))   # soxr_hq clone
SCI = firwin(81,  0.25,          window=('kaiser',5.0))     # == scipy resample_poly default for 4:3 / 3:4
def emit(name, h, hl, f):
    f.write(f"    /// {len(h)} taps, half_len = {hl}, unit DC gain (sum = {h.sum():.9f}).\n")
    f.write(f"    static let {name}: [Double] = [\n")
    for i in range(0, len(h), 4):
        f.write("        " + ", ".join(f"{v: .17e}" for v in h[i:i+4]) + ",\n")
    f.write("    ]\n\n")
with open("/tmp/PerthResamplerTaps.swift","w") as f:
    f.write("// Generated. Kaiser-windowed sinc prototypes on the 96 kHz (= 4x24k = 3x32k) grid.\n")
    f.write("enum PerthResamplerTaps {\n")
    emit("hq", HQ, 400, f)          # fc=11482 Hz, beta=13.75
    emit("scipyPolyphase", SCI, 40, f)
    f.write("}\n")
print("wrote /tmp/PerthResamplerTaps.swift", len(HQ), len(SCI))
np.save("/tmp/hq801.npy", HQ)
