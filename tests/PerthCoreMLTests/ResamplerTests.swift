import Foundation
import Testing

@testable import PerthCoreML

struct ResampleFixture: Decodable {
    struct PolyCase: Decodable {
        let n: Int
        let x: String
        let y: String
    }
    struct HQCase: Decodable {
        let n: Int
        let x: String
        let y: String
        let pythonLastIsZero: Bool
    }
    struct DownCase: Decodable {
        let n: Int
        let x: String
        let y: String
    }
    let polyUp24to32: [PolyCase]
    let hqUp24to32: [HQCase]
    let hqDown32to24: [DownCase]
}

@Suite("Resampler")
struct ResamplerTests {

    @Test("output length is ceil(n * up / down), as scipy and librosa both are")
    func outputLength() {
        for n in 1...400 {
            #expect(Resampler.outputLength(n, up: 4, down: 3) == (n * 4 + 2) / 3, "n=\(n)")
            #expect(Resampler.outputLength(n, up: 3, down: 4) == (n * 3 + 3) / 4, "n=\(n)")
        }
        #expect(Resampler.outputLength(24001, up: 4, down: 3) == 32002)
        #expect(Resampler.outputLength(24000, up: 4, down: 3) == 32000)
    }

    /// The detect path. Python calls `librosa.resample(res_type: "polyphase")`, which is scipy's
    /// `resample_poly` -- and the detector's answer depends on it: swapping in the HQ filter moves
    /// the magmask threshold enough to flip mask bits.
    @Test("polyUp24to32 matches scipy.signal.resample_poly")
    func polyMatchesScipy() throws {
        let fx: ResampleFixture = try Fixture.load("resample")
        #expect(!fx.polyUp24to32.isEmpty)

        for c in fx.polyUp24to32 {
            let got = Resampler.up24to32Detect(Fixture.f32(c.x))
            let want = Fixture.f32(c.y)
            #expect(got.count == want.count, "n=\(c.n)")
            guard got.count == want.count else { continue }
            let d = maxAbsDiff(got, want)
            #expect(d < 1e-6, "n=\(c.n): max|d| = \(d) vs scipy")
        }
    }

    /// The apply path. Python calls `librosa.resample` with its default `res_type`, i.e. soxr_hq.
    /// We do not vendor libsoxr; an 801-tap Kaiser firwin stands in for it, so this is an SNR gate
    /// and not a bit-exactness one.
    @Test("hqUp24to32 reproduces librosa's soxr_hq")
    func hqUpMatchesLibrosa() throws {
        let fx: ResampleFixture = try Fixture.load("resample")
        #expect(!fx.hqUp24to32.isEmpty)

        for c in fx.hqUp24to32 {
            let got = Resampler.up24to32Apply(Fixture.f32(c.x))
            let want = Fixture.f32(c.y)
            #expect(got.count == want.count, "n=\(c.n)")
            guard got.count == want.count else { continue }

            #expect(interiorSNR(want, got) > 120,
                    "n=\(c.n): steady-state SNR \(interiorSNR(want, got)) dB vs librosa")
            if c.n >= 6000 {
                let db = snr(want, got)
                #expect(db > 100, "n=\(c.n): SNR \(db) dB vs librosa")
            }
        }
    }

    /// soxr's own output is `round(n*4/3)` and librosa's `fix_length` stretches that to the ceil
    /// **by appending a zero** -- which happens only when the fraction is below a half, i.e. when
    /// `n % 3 == 1`. `n % 3 == 2` rounds up instead and ends on a real sample.
    @Test("the trailing sample follows librosa's fix_length")
    func trailingSample() throws {
        let fx: ResampleFixture = try Fixture.load("resample")

        for c in fx.hqUp24to32 {
            let got = Resampler.up24to32Apply(Fixture.f32(c.x))
            let want = Fixture.f32(c.y)
            guard let gotLast = got.last, let wantLast = want.last else {
                Issue.record("n=\(c.n): empty output")
                continue
            }
            #expect(c.pythonLastIsZero == (c.n % 3 == 1),
                    "n=\(c.n): librosa zero-fills iff n % 3 == 1")

            if c.n % 3 == 1 {
                #expect(wantLast == 0, "n=\(c.n): librosa's own last sample")
                #expect(gotLast == 0, "n=\(c.n): expected an appended zero, got \(gotLast)")
            } else {
                #expect(wantLast != 0, "n=\(c.n): librosa's own last sample")
                #expect(gotLast != 0,
                        "n=\(c.n): librosa keeps a real sample here, we emitted \(gotLast)")
                #expect(abs(gotLast - wantLast) < 1e-3, "n=\(c.n): last sample \(gotLast) vs \(wantLast)")
            }
        }
    }

    /// `down32to24Apply` is only ever handed an ISTFT output, which is a whole number of 320-sample
    /// hops -- so it can never land on librosa's zero-fill (4 divides 320).
    @Test("hqDown32to24 reproduces librosa's soxr_hq on hop-multiple lengths")
    func hqDownMatchesLibrosa() throws {
        let fx: ResampleFixture = try Fixture.load("resample")
        #expect(!fx.hqDown32to24.isEmpty)

        for c in fx.hqDown32to24 {
            #expect(c.n % PerthConfig.hopSize == 0, "n=\(c.n) is not a hop multiple")
            let got = Resampler.down32to24Apply(Fixture.f32(c.x))
            let want = Fixture.f32(c.y)
            #expect(got.count == want.count, "n=\(c.n)")
            guard got.count == want.count else { continue }

            #expect(interiorSNR(want, got) > 120,
                    "n=\(c.n): steady-state SNR \(interiorSNR(want, got)) dB vs librosa")
            let db = snr(want, got)
            #expect(db > 100, "n=\(c.n): SNR \(db) dB vs librosa")
        }
    }

    @Test("an empty signal resamples to an empty signal")
    func empty() {
        #expect(Resampler.up24to32Apply([]).isEmpty)
        #expect(Resampler.up24to32Detect([]).isEmpty)
        #expect(Resampler.down32to24Apply([]).isEmpty)
    }

    /// SNR outside the filter's edge transient.
    ///
    /// soxr primes its own delay line at the signal boundaries and our firwin does not reproduce
    /// that, so the first and last few hundred output samples disagree by up to 1e-5 while the
    /// steady state agrees to ~133 dB. On a 43 ms clip that transient is a third of the signal and
    /// drags the whole-signal SNR under 100 dB; on anything of a realistic length it does not.
    /// Gating on the steady state is what makes the short cases say something.
    private func interiorSNR(_ ref: [Float], _ got: [Float]) -> Double {
        let edge = 420                                   // the 801-tap prototype's half-length
        guard ref.count > 2 * edge + 16 else { return .infinity }
        let r = ref.count - edge
        return snr(Array(ref[edge..<r]), Array(got[edge..<r]))
    }
}
