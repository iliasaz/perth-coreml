import Accelerate
import Foundation

/// Polyphase FIR resampling between 24 kHz and 32 kHz (an exact 4:3 ratio).
///
/// Two filter sets, because Python uses two:
///   - `apply_watermark` resamples with librosa's DEFAULT `res_type`, which is `soxr_hq`.
///   - `get_watermark` resamples with `res_type: "polyphase"`, i.e. scipy's `resample_poly`.
///
/// We reproduce each with its own tap table (see `PerthAssets`). Using the HQ filter on the detect
/// path would be a bug, not an upgrade: its different stopband shifts the `magmask` threshold
/// (the dB transform amplifies stopband differences across all 1025 bins) and flips mask bits --
/// and the gate we care about is "our detector agrees with Python's detector".
enum Resampler {

    /// scipy's `resample_poly` output length, which is also librosa's: `ceil(n * up / down)`.
    static func outputLength(_ n: Int, up: Int, down: Int) -> Int {
        (n * up + down - 1) / down
    }

    /// `upfirdn(taps, x, up, down)` followed by scipy's `preRemove` trim.
    ///
    /// Each output sample is a decimating FIR over one polyphase residue class:
    ///     p = down*m + 0 ; phase = p % up ; base = p / up
    ///     y[m] = sum_i taps[phase + up*i] * x[base - i]        (x is zero outside [0, n))
    /// The `preRemove` offset scipy applies to undo the filter's group delay is folded into `p`.
    static func resample(_ x: [Float], _ f: PolyphaseTaps) -> [Float] {
        let n = x.count
        guard n > 0 else { return [] }
        let nOut = outputLength(n, up: f.up, down: f.down)
        let taps = f.taps
        let nTaps = taps.count

        var out = [Float](repeating: 0, count: nOut)
        x.withUnsafeBufferPointer { xp in
            taps.withUnsafeBufferPointer { hp in
                for m in 0..<nOut {
                    let p = f.down * (m + f.preRemove)
                    let phase = p % f.up
                    let base = p / f.up
                    var acc = 0.0
                    var i = 0
                    // taps[phase + up*i] pairs with x[base - i]; both ranges are clamped.
                    let iMax = min((nTaps - 1 - phase) / f.up, base)
                    let iMin = max(0, base - n + 1)
                    i = iMin
                    while i <= iMax {
                        acc += hp[phase + f.up * i] * Double(xp[base - i])
                        i += 1
                    }
                    out[m] = Float(acc)
                }
            }
        }
        return out
    }

    /// 24 kHz -> 32 kHz on the apply path (soxr_hq-equivalent).
    ///
    /// The trailing-zero quirk is real and must be reproduced: soxr's own output is
    /// `round(n*4/3)`, but librosa asks `fix_length` for `ceil(n*4/3)` and makes up the shortfall
    /// **by appending a zero**. So the zero appears only when the exact ratio rounds DOWN, i.e.
    /// when `n % 3 == 1` -- a third of all input lengths. `n % 3 == 2` rounds UP and ends on a
    /// real sample; zeroing it too would put a 6e-3 error in the last sample.
    static func up24to32Apply(_ x: [Float]) -> [Float] {
        var y = resample(x, PerthAssets.hqUp24to32)
        if roundsDown(x.count, up: 4, down: 3), let last = y.indices.last {
            y[last] = 0          // librosa's fix_length pads with a zero, not with a real sample
        }
        return y
    }

    /// Whether `n * up / down` rounds down, which is exactly when librosa's `fix_length` has to
    /// append a zero to reach the `ceil` it asked soxr for. soxr rounds halves UP, so an exact
    /// half is not a shortfall.
    private static func roundsDown(_ n: Int, up: Int, down: Int) -> Bool {
        let rem = (n * up) % down
        return rem != 0 && 2 * rem < down
    }

    /// 32 kHz -> 24 kHz on the apply path (soxr_hq-equivalent).
    ///
    /// Cannot hit the zero-pad quirk: the 32 kHz signal handed back by the ISTFT is always a
    /// multiple of the hop (320), and 4 divides 320.
    static func down32to24Apply(_ x: [Float]) -> [Float] {
        resample(x, PerthAssets.hqDown32to24)
    }

    /// 24 kHz -> 32 kHz on the detect path (scipy `resample_poly`-equivalent).
    static func up24to32Detect(_ x: [Float]) -> [Float] {
        resample(x, PerthAssets.polyUp24to32)
    }
}
