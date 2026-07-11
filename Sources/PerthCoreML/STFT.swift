import Accelerate
import Foundation

/// torchaudio's `Spectrogram(power: nil, center: true)` / `InverseSpectrogram`, in fp32.
///
/// Not thread-safe: it owns scratch buffers. Use one instance per thread.
///
/// vDSP packing (measured on this machine, not recalled):
///   - `vDSP_fft_zrip` forward returns exactly **2x** the true DFT -- scale by 0.5.
///   - The **Nyquist bin's real part lives in `imagp[0]`**, not in a 1025th slot; DC is in
///     `realp[0]`. Both have zero imaginary part.
///   - Inverse: pack the true (already halved) spectrum, run `zrip` inverse, `ztoc`, then scale
///     by `1/nFFT`.
final class STFT {
    private let nFFT: Int
    private let hop: Int
    private let nFreq: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private let window: [Float]
    private let windowSq: [Float]

    /// Scratch. Sized once; reused across frames.
    private var realp: [Float]
    private var imagp: [Float]
    private var frame: [Float]

    init(window: [Float], nFFT: Int = PerthConfig.nFFT, hop: Int = PerthConfig.hopSize) {
        precondition(window.count == nFFT)
        precondition(nFFT.nonzeroBitCount == 1, "vDSP radix-2 FFT needs a power-of-two size")
        self.nFFT = nFFT
        self.hop = hop
        self.nFreq = nFFT / 2 + 1
        self.log2n = vDSP_Length(nFFT.trailingZeroBitCount)
        guard let s = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            preconditionFailure("vDSP_create_fftsetup failed for n=\(nFFT)")
        }
        self.setup = s
        self.window = window
        self.windowSq = window.map { $0 * $0 }
        self.realp = [Float](repeating: 0, count: nFFT / 2)
        self.imagp = [Float](repeating: 0, count: nFFT / 2)
        self.frame = [Float](repeating: 0, count: nFFT)
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    /// Number of STFT frames for a signal of `n` samples, with `center: true`.
    static func frameCount(_ n: Int, hop: Int = PerthConfig.hopSize) -> Int { n / hop + 1 }

    /// The signal length `inverse` returns for `t` frames.
    ///
    /// Note this is generally SHORTER than the signal that produced the spectrogram: torch's
    /// istft with `length: nil` yields `nFFT + hop*(t-1)` samples and trims `nFFT/2` from each
    /// end. Perth never passes `length:`, so `apply_watermark` hands back up to `hop-1` fewer
    /// samples than it was given. We reproduce that rather than "fixing" it.
    static func signalLength(frames t: Int, hop: Int = PerthConfig.hopSize) -> Int {
        max(0, hop * (t - 1))
    }

    /// Reflect-pad by `nFFT/2` each side, without repeating the edge sample -- numpy's
    /// `mode="reflect"`, which is what `center: true` uses.
    private func reflectPad(_ x: [Float]) -> [Float] {
        let pad = nFFT / 2
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }
        for i in 0..<n { out[pad + i] = x[i] }
        for i in 0..<pad { out[pad + n + i] = x[n - 2 - i] }
        return out
    }

    /// Signal -> (magnitude in dB-normalised units, phase in radians), both `(nFreq, T)`
    /// laid out frequency-major: index `k * T + t`.
    func forward(_ signal: [Float]) throws -> (mag: [Float], phase: [Float], frames: Int) {
        guard signal.count >= PerthConfig.minSamples32k else {
            throw PerthError.signalTooShort(samples: signal.count,
                                            minimum: PerthConfig.minSamples32k)
        }
        let t = STFT.frameCount(signal.count, hop: hop)
        let padded = reflectPad(signal)

        var mag = [Float](repeating: 0, count: nFreq * t)
        var phase = [Float](repeating: 0, count: nFreq * t)
        var re = [Float](repeating: 0, count: nFreq)
        var im = [Float](repeating: 0, count: nFreq)

        for f in 0..<t {
            vDSP_vmul(Array(padded[(f * hop)..<(f * hop + nFFT)]), 1, window, 1, &frame, 1,
                      vDSP_Length(nFFT))
            spectrum(of: frame, re: &re, im: &im)

            // mag = |X|, clipped at 1e-9 BEFORE log10 -- digital silence would otherwise be -inf.
            // phase = atan2(im, re).
            var m = [Float](repeating: 0, count: nFreq)
            var p = [Float](repeating: 0, count: nFreq)
            var cnt = Int32(nFreq)
            vDSP_vdist(re, 1, im, 1, &m, 1, vDSP_Length(nFreq))     // hypot(re, im)
            vvatan2f(&p, im, re, &cnt)                              // atan2(y=im, x=re)

            var floorV = PerthConfig.stftMagnitudeMin
            vDSP_vthr(m, 1, &floorV, &m, 1, vDSP_Length(nFreq))     // max(m, 1e-9)
            var log10m = [Float](repeating: 0, count: nFreq)
            vvlog10f(&log10m, m, &cnt)
            // db = 20*log10(m); norm = (db - minLevelDB) / denormScale
            var a: Float = 20 / PerthConfig.denormScale
            var b: Float = -PerthConfig.minLevelDB / PerthConfig.denormScale
            vDSP_vsmsa(log10m, 1, &a, &b, &log10m, 1, vDSP_Length(nFreq))

            for k in 0..<nFreq {
                mag[k * t + f] = log10m[k]
                phase[k * t + f] = p[k]
            }
        }
        return (mag, phase, t)
    }

    /// (dB-normalised magnitude, phase) -> signal of `hop*(T-1)` samples.
    func inverse(mag: [Float], phase: [Float], frames t: Int) -> [Float] {
        let total = nFFT + hop * (t - 1)
        var acc = [Float](repeating: 0, count: total)
        var env = [Float](repeating: 0, count: total)

        var re = [Float](repeating: 0, count: nFreq)
        var im = [Float](repeating: 0, count: nFreq)
        var lin = [Float](repeating: 0, count: nFreq)
        var sinp = [Float](repeating: 0, count: nFreq)
        var cosp = [Float](repeating: 0, count: nFreq)
        var col = [Float](repeating: 0, count: nFreq)
        var colPhase = [Float](repeating: 0, count: nFreq)

        for f in 0..<t {
            for k in 0..<nFreq {
                col[k] = mag[k * t + f]
                colPhase[k] = phase[k * t + f]
            }
            // db = norm*195 - 180;  lin = 10 ** min(db/20, 10)   [torch: .clip(max: 10)]
            var a = PerthConfig.denormScale / 20
            var b = PerthConfig.minLevelDB / 20
            vDSP_vsmsa(col, 1, &a, &b, &col, 1, vDSP_Length(nFreq))
            // vDSP_vthr is a lower bound only, so clamp from above by negating twice.
            vDSP_vneg(col, 1, &col, 1, vDSP_Length(nFreq))
            var negHi: Float = -10
            vDSP_vthr(col, 1, &negHi, &col, 1, vDSP_Length(nFreq))   // max(-db/20, -10)
            vDSP_vneg(col, 1, &col, 1, vDSP_Length(nFreq))           // => min(db/20, 10)

            var cnt = Int32(nFreq)
            let base = [Float](repeating: 10, count: nFreq)
            vvpowf(&lin, col, base, &cnt)                            // z = base ** exponent
            vvsincosf(&sinp, &cosp, colPhase, &cnt)
            vDSP_vmul(lin, 1, cosp, 1, &re, 1, vDSP_Length(nFreq))
            vDSP_vmul(lin, 1, sinp, 1, &im, 1, vDSP_Length(nFreq))

            inverseSpectrum(re: re, im: im, into: &frame)

            // The window is applied a SECOND time on the way out, then overlap-added; the
            // window-square envelope is accumulated alongside and divided out at the end.
            var out = [Float](repeating: 0, count: nFFT)
            vDSP_vmul(frame, 1, window, 1, &out, 1, vDSP_Length(nFFT))
            let off = f * hop
            for i in 0..<nFFT {
                acc[off + i] += out[i]
                env[off + i] += windowSq[i]
            }
        }

        // The envelope is NOT constant: 2048/320 = 6.4 is not an integer, so hann^2 does not
        // satisfy COLA. Dividing by a constant Sum(w^2)/hop would put a ~29% amplitude error in
        // the first and last 1024 samples of every utterance. Divide by the real envelope.
        let pad = nFFT / 2
        let n = STFT.signalLength(frames: t, hop: hop)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let e = env[pad + i]
            out[i] = e > 1e-11 ? acc[pad + i] / e : 0
        }
        return out
    }

    // MARK: - vDSP real FFT

    private func spectrum(of x: [Float], re: inout [Float], im: inout [Float]) {
        let half = nFFT / 2
        realp.withUnsafeMutableBufferPointer { rp in
            imagp.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                x.withUnsafeBufferPointer { xp in
                    xp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                        vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                // zrip returns 2x the true DFT. DC in realp[0], Nyquist in imagp[0].
                re[0] = rp[0] * 0.5
                im[0] = 0
                re[half] = ip[0] * 0.5
                im[half] = 0
                for k in 1..<half {
                    re[k] = rp[k] * 0.5
                    im[k] = ip[k] * 0.5
                }
            }
        }
    }

    private func inverseSpectrum(re: [Float], im: [Float], into out: inout [Float]) {
        let half = nFFT / 2
        realp.withUnsafeMutableBufferPointer { rp in
            imagp.withUnsafeMutableBufferPointer { ip in
                // Repack DC and Nyquist into slot 0; their imaginary parts are discarded, which
                // is exactly what torch does (verified delta == 0).
                rp[0] = re[0]
                ip[0] = re[half]
                for k in 1..<half {
                    rp[k] = re[k]
                    ip[k] = im[k]
                }
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_INVERSE))
                out.withUnsafeMutableBufferPointer { op in
                    op.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                        vDSP_ztoc(&split, 1, $0, 2, vDSP_Length(half))
                    }
                }
            }
        }
        var scale = Float(1) / Float(nFFT)
        vDSP_vsmul(out, 1, &scale, &out, 1, vDSP_Length(nFFT))
    }
}

extension PerthConfig {
    static let stftMagnitudeMin: Float = 1e-9
}
