import Accelerate
import Foundation

enum Spectral {
    /// Frames whose total energy across ALL 1025 bins exceeds 5% of the loudest frame.
    ///
    /// This cannot live inside the CoreML graph: the model only ever sees the low 128 bins, and
    /// the threshold is a global max over the whole (unpadded) signal.
    ///
    /// `mag` is `(nFreq, T)` frequency-major. Returns `T` values, each 1.0 or 0.0.
    ///
    /// Two details that decide bit-exactness against torch:
    ///   - accumulate the per-frame sum in Double (torch's blocked fp32 reduction is *less*
    ///     accurate; Double lands within a couple of ULP of it and never further away);
    ///   - compute the threshold as `Float(max) * Float(0.05)`, NOT promoted to Double --
    ///     the fp32 product matches torch in 20000/20000 trials, the Double-promoted one
    ///     matches only 15988/20000.
    ///   - the comparison is a STRICT `>`.
    static func magmask(_ mag: [Float], frames t: Int, nFreq: Int = PerthConfig.nFreq) -> [Float] {
        var sums = [Float](repeating: 0, count: t)
        for f in 0..<t {
            var acc = 0.0
            for k in 0..<nFreq { acc += Double(mag[k * t + f]) }
            sums[f] = Float(acc)
        }
        let peak = sums.max() ?? 0
        let threshold = peak * PerthConfig.magmaskP
        return sums.map { $0 > threshold ? 1 : 0 }
    }

    /// How close the quietest kept frame is to the magmask threshold, in units of the worst
    /// plausible numeric error. Small values mean a mask bit is one rounding away from flipping,
    /// which would change the detector's answer. Callers can log this as a CI alarm.
    static func magmaskMargin(_ mag: [Float], frames t: Int,
                              nFreq: Int = PerthConfig.nFreq) -> Float {
        var sums = [Float](repeating: 0, count: t)
        for f in 0..<t {
            var acc = 0.0
            for k in 0..<nFreq { acc += Double(mag[k * t + f]) }
            sums[f] = Float(acc)
        }
        let threshold = (sums.max() ?? 0) * PerthConfig.magmaskP
        return sums.map { abs($0 - threshold) }.min() ?? 0
    }
}
