import Accelerate
import Foundation

/// Rational-rate polyphase FIR resampler, bit-compatible with
/// `scipy.signal.resample_poly` (and therefore with `librosa.resample(res_type="polyphase")`).
///
/// Model:  x --(up-sample by `up`, zero-stuff)--> h[] --(decimate by `down`)--> y
/// where h is a symmetric, linear-phase, unit-DC-gain lowpass prototype on the
/// (up * inRate) grid, scaled by `up` to preserve gain through zero-stuffing.
///
/// The zero-phase alignment scipy achieves via (n_pre_pad, n_pre_remove) reduces
/// EXACTLY to a constant offset of `halfLen` on the up-sampled grid:
///
///     p     = down * m + halfLen        // position on the up-sampled grid
///     phase = p % up
///     base  = p / up
///     y[m]  = sum_i  h[phase + up*i] * x[base - i]        (x = 0 outside [0, n))
///
/// Output length is `ceil(n * up / down)`, matching both scipy and librosa.
public struct PolyphaseResampler: Sendable {
    public let up: Int
    public let down: Int
    private let tapsPerPhase: Int
    /// phaseFilters[phase] holds the sub-filter for that phase, REVERSED and
    /// zero-padded to `tapsPerPhase`, so each output is a straight forward dot product.
    private let phaseFilters: [[Float]]
    /// For output residue r (m ≡ r mod up): the (constant) phase and the base offset.
    private let residuePhase: [Int]
    private let residueBase: [Int]

    /// - Parameters:
    ///   - prototype: symmetric lowpass FIR with unit DC gain, length 2*halfLen+1.
    ///   - halfLen: (count - 1) / 2.
    public init(up: Int, down: Int, prototype: [Double], halfLen: Int) {
        precondition(up >= 1 && down >= 1)
        precondition(prototype.count == 2 * halfLen + 1, "prototype must be 2*halfLen+1 long")
        let g = gcd(up, down)
        let u = up / g, d = down / g
        self.up = u
        self.down = d

        let n = prototype.count
        let lp = (n + u - 1) / u                    // ceil(n / up)
        self.tapsPerPhase = lp

        // Scale by `up` (compensates zero-stuffing), split into phases, reverse each.
        var filters = [[Float]](repeating: [Float](repeating: 0, count: lp), count: u)
        for phase in 0..<u {
            var sub = [Float](repeating: 0, count: lp)
            var i = 0
            while phase + u * i < n {
                sub[i] = Float(prototype[phase + u * i] * Double(u))
                i += 1
            }
            filters[phase] = sub.reversed()          // pre-reverse => forward dot product
        }
        self.phaseFilters = filters

        var rp = [Int](repeating: 0, count: u)
        var rb = [Int](repeating: 0, count: u)
        for r in 0..<u {
            let p = d * r + halfLen
            let phase = p % u
            rp[r] = phase
            rb[r] = (p - phase) / u                  // exact
        }
        self.residuePhase = rp
        self.residueBase = rb
    }

    /// Output length for an input of `n` samples: ceil(n * up / down).
    public func outputCount(forInputCount n: Int) -> Int {
        n <= 0 ? 0 : (n * up + down - 1) / down
    }

    public func resample(_ x: [Float]) -> [Float] {
        let nIn = x.count
        let nOut = outputCount(forInputCount: nIn)
        guard nOut > 0 else { return [] }
        let lp = tapsPerPhase

        // Zero-pad so every tap read is in bounds.
        // Signal index j maps to xpad index j + lp.
        // Window for output m starts at signal index (base - lp + 1) => xpad index (base + 1).
        // base_max <= (down*(nOut-1) + halfLen)/up  <  nIn + lp, so `lp + down` of right pad suffices.
        let pad = lp + down
        var xpad = [Float](repeating: 0, count: pad + nIn + pad)
        xpad.withUnsafeMutableBufferPointer { dst in
            x.withUnsafeBufferPointer { src in
                if nIn > 0 { dst.baseAddress!.advanced(by: pad).update(from: src.baseAddress!, count: nIn) }
            }
        }

        var y = [Float](repeating: 0, count: nOut)
        y.withUnsafeMutableBufferPointer { yb in
            xpad.withUnsafeBufferPointer { xb in
                for r in 0..<up {
                    // outputs m = up*t + r, t = 0 ..< nT
                    let nT = (nOut - r + up - 1) / up
                    if nT <= 0 { continue }
                    let phase = residuePhase[r]
                    let base0 = residueBase[r]
                    // xpad start for t: down*t + (base0 + 1) + lp   [+lp = signal->xpad offset]
                    let start = base0 + 1 + pad - lp
                    var tmp = [Float](repeating: 0, count: nT)
                    phaseFilters[phase].withUnsafeBufferPointer { fb in
                        // C[i] = sum_k A[i*down + k] * F[k]
                        vDSP_desamp(xb.baseAddress!.advanced(by: start),
                                    down,
                                    fb.baseAddress!,
                                    &tmp,
                                    vDSP_Length(nT),
                                    vDSP_Length(lp))
                    }
                    // scatter: y[up*t + r] = tmp[t]
                    cblas_scopy(Int32(nT), tmp, 1, yb.baseAddress!.advanced(by: r), Int32(up))
                }
            }
        }
        return y
    }
}

@inline(__always) func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

// Convenience: the two rates Perth needs.
public enum PerthResampler {
    /// soxr_hq-equivalent (clean-room Kaiser-windowed sinc, fc = 11482 Hz on the 96 kHz grid, beta = 13.75).
    public static let up24to32 = PolyphaseResampler(up: 4, down: 3, prototype: PerthResamplerTaps.hq, halfLen: 400)
    public static let down32to24 = PolyphaseResampler(up: 3, down: 4, prototype: PerthResamplerTaps.hq, halfLen: 400)
    /// Bit-compatible with librosa res_type="polyphase" (scipy default). Test/parity use.
    public static let up24to32Scipy = PolyphaseResampler(up: 4, down: 3, prototype: PerthResamplerTaps.scipyPolyphase, halfLen: 40)
    public static let down32to24Scipy = PolyphaseResampler(up: 3, down: 4, prototype: PerthResamplerTaps.scipyPolyphase, halfLen: 40)
}
