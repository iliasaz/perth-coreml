import Foundation
import Testing

@testable import PerthCoreML

struct STFTFixture: Decodable {
    struct Case: Decodable {
        let name: String
        let n: Int
        let frames: Int
        let signal: String
        let mag: String
        let phase: String
        let istft: String
    }
    let nFFT: Int
    let hop: Int
    let nFreq: Int
    let cases: [Case]
}

@Suite("STFT / ISTFT")
struct STFTTests {
    let stft = STFT(window: PerthAssets.stftWindow)

    @Test("frame count is N/hop + 1")
    func frameCount() {
        for n in [1025, 1280, 3201, 6400, 16000, 32000, 96137] {
            #expect(STFT.frameCount(n) == n / PerthConfig.hopSize + 1, "n=\(n)")
        }
    }

    @Test("forward produces T frames and inverse returns exactly hop*(T-1) samples")
    func lengths() throws {
        for n in [1025, 1280, 3201, 6400, 16000] {
            let x = whiteNoise(n, seed: UInt64(n))
            let (mag, phase, t) = try stft.forward(x)
            #expect(t == STFT.frameCount(n), "n=\(n)")
            #expect(mag.count == PerthConfig.nFreq * t, "n=\(n)")
            #expect(phase.count == PerthConfig.nFreq * t, "n=\(n)")

            let y = stft.inverse(mag: mag, phase: phase, frames: t)
            #expect(y.count == PerthConfig.hopSize * (t - 1), "n=\(n)")
            // The ISTFT is called without `length:`, so it hands back up to hop-1 fewer samples
            // than it was given. That truncation is stock Perth's, not a bug to fix here.
            #expect(y.count <= n, "n=\(n)")
            #expect(n - y.count < PerthConfig.hopSize, "n=\(n)")
        }
    }

    @Test("round trip reconstructs the signal it was given")
    func roundTrip() throws {
        for n in [1025, 3201, 6400, 16000] {
            let x = whiteNoise(n, seed: UInt64(n) &* 7)
            let (mag, phase, t) = try stft.forward(x)
            let y = stft.inverse(mag: mag, phase: phase, frames: t)

            // Everything the ISTFT returns is comparable: `acc[j] == xp[j] * env[j]` exactly, for
            // every j, whichever frames happen to overlap it -- so the only lossy step is the
            // magnitude's trip through log10/10**, and the reconstruction has no bad edge.
            let ref = Array(x[0..<y.count])
            let db = snr(ref, y)
            #expect(db > 100, "n=\(n): round-trip SNR \(db) dB")
        }
    }

    @Test("digital silence stays exactly silent")
    func silence() throws {
        let x = [Float](repeating: 0, count: 6400)
        let (mag, phase, t) = try stft.forward(x)

        // Torch normalises silence to exactly 0.0 (20*log10(1e-9) == -180 to the bit). vvlog10f
        // returns -9.000001 rather than -9, so this lands one ULP below instead -- harmless, but
        // the exactness the silence guarantee needs is UNIFORMITY across bins, not zero: a
        // constant spectrum is what makes the irfft a clean delta at sample 0.
        #expect(Set(mag.map(\.bitPattern)).count == 1, "silence must normalise to one value")
        #expect(abs(mag[0]) < 1e-6, "normalised magnitude of silence = \(mag[0])")
        #expect(phase.allSatisfy { $0 == 0 }, "atan2(0, 0) is 0")

        let y = stft.inverse(mag: mag, phase: phase, frames: t)
        // Silence survives only because window[0] == 0: every bin denormalises to ~1e-9, the irfft
        // of a constant spectrum is a delta at sample 0, and the window zeroes it. An ISTFT that
        // windows in a different order returns 1e-9 noise here instead.
        #expect(y.count == PerthConfig.hopSize * (t - 1))
        #expect(y.allSatisfy { $0 == 0 }, "max |y| = \(y.map(abs).max() ?? 0), expected exact 0")
    }

    @Test("a signal shorter than the reflect pad throws")
    func tooShort() {
        for n in [0, 1, 512, 1024] {
            #expect(throws: PerthError.self, "n=\(n)") {
                _ = try STFT(window: PerthAssets.stftWindow).forward(whiteNoise(n, seed: 1))
            }
        }
        // 1025 is the first length that mirrors.
        #expect(throws: Never.self) {
            _ = try STFT(window: PerthAssets.stftWindow).forward(whiteNoise(1025, seed: 1))
        }
    }

    @Test("signalTooShort reports the sample count and the minimum")
    func tooShortPayload() {
        do {
            _ = try stft.forward(whiteNoise(1024, seed: 1))
            Issue.record("expected a throw")
        } catch let PerthError.signalTooShort(samples, minimum) {
            #expect(samples == 1024)
            #expect(minimum == PerthConfig.minSamples32k)
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test("matches torchaudio Spectrogram / InverseSpectrogram")
    func matchesTorch() throws {
        let fx: STFTFixture = try Fixture.load("stft")
        #expect(fx.nFFT == PerthConfig.nFFT && fx.hop == PerthConfig.hopSize)
        #expect(!fx.cases.isEmpty)

        for c in fx.cases {
            let wantMag = Fixture.f32(c.mag)
            let wantPhase = Fixture.f32(c.phase)
            let wantOut = Fixture.f32(c.istft)

            let (mag, phase, t) = try stft.forward(Fixture.f32(c.signal))
            #expect(t == c.frames, "\(c.name)")
            #expect(maxAbsDiff(mag, wantMag) < 1e-5,
                    "\(c.name): max|dmag| = \(maxAbsDiff(mag, wantMag))")

            // Phase alone is the wrong thing to bound: a bin whose magnitude sits near the FFT's
            // absolute error floor has an essentially arbitrary angle, and comparing it measures
            // pocketfft's round-off, not the port. Bound the error in the complex plane instead,
            // relative to the loudest bin in the same frame -- that is what the ISTFT sees, and a
            // wrong FFT scale or a swapped atan2 argument moves it by O(1).
            let err = worstComplexError(mag: mag, phase: phase,
                                        wantMag: wantMag, wantPhase: wantPhase, frames: t)
            #expect(err < 1e-5, "\(c.name): worst complex error, relative to the frame peak = \(err)")

            let y = stft.inverse(mag: mag, phase: phase, frames: t)
            #expect(y.count == wantOut.count, "\(c.name)")
            let db = snr(wantOut, y)
            #expect(db > 100, "\(c.name): ISTFT vs torch SNR \(db) dB")
        }
    }

    /// `max_k |X - X_torch| / max_k |X_torch|`, worst over frames. `X` is rebuilt from the
    /// (magnitude, phase) pair exactly as `inverse` rebuilds it.
    private func worstComplexError(mag: [Float], phase: [Float],
                                   wantMag: [Float], wantPhase: [Float], frames t: Int) -> Float {
        func linear(_ norm: Float) -> Float {
            powf(10, Swift.min(norm * PerthConfig.denormScale + PerthConfig.minLevelDB, 200) / 20)
        }
        var worst: Float = 0
        for f in 0..<t {
            var peak: Float = 0
            var err: Float = 0
            for k in 0..<PerthConfig.nFreq {
                let i = k * t + f
                let a = linear(mag[i])
                let b = linear(wantMag[i])
                let dRe = a * cosf(phase[i]) - b * cosf(wantPhase[i])
                let dIm = a * sinf(phase[i]) - b * sinf(wantPhase[i])
                peak = Swift.max(peak, b)
                err = Swift.max(err, (dRe * dRe + dIm * dIm).squareRoot())
            }
            if peak > 0 { worst = Swift.max(worst, err / peak) }
        }
        return worst
    }
}
