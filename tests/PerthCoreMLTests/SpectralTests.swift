import Foundation
import Testing

@testable import PerthCoreML

@Suite("magmask")
struct SpectralTests {

    /// The comparison is a STRICT `>`. Digital silence is the case that proves it: every frame sums
    /// to 0, the threshold is 0, and `>=` would light up the entire mask instead of clearing it.
    @Test("an all-silent signal produces an all-zero mask")
    func silence() throws {
        let stft = STFT(window: PerthAssets.stftWindow)
        let (mag, _, t) = try stft.forward([Float](repeating: 0, count: 6400))

        let mask = Spectral.magmask(mag, frames: t)
        #expect(mask.count == t)
        #expect(mask.allSatisfy { $0 == 0 }, "\(mask.filter { $0 != 0 }.count) frames wrongly kept")
    }

    @Test("a frame sitting exactly on the threshold is dropped")
    func strictComparison() {
        // 32 * Float(0.05) is exact (multiplying by a power of two only moves the exponent), so a
        // frame can be placed precisely ON the threshold rather than near it.
        let peak: Float = 32
        let threshold = peak * PerthConfig.magmaskP
        let nFreq = 4
        let t = 4
        let sums: [Float] = [peak, threshold, threshold.nextUp, threshold.nextDown]

        var mag = [Float](repeating: 0, count: nFreq * t)
        for f in 0..<t { mag[0 * t + f] = sums[f] }      // all of a frame's energy in bin 0

        let mask = Spectral.magmask(mag, frames: t, nFreq: nFreq)
        #expect(mask == [1, 0, 1, 0], "exactly-at-threshold must be dropped, got \(mask)")
    }

    @Test("the mask keeps loud frames and drops quiet ones")
    func gating() {
        let nFreq = 8
        let t = 3
        var mag = [Float](repeating: 0, count: nFreq * t)
        for k in 0..<nFreq {
            mag[k * t + 0] = 1.0        // loud
            mag[k * t + 1] = 0.01       // 1 % of the peak: below the 5 % threshold
            mag[k * t + 2] = 0.5
        }
        #expect(Spectral.magmask(mag, frames: t, nFreq: nFreq) == [1, 0, 1])
    }

    @Test("the margin reports the closest frame's distance to the threshold")
    func margin() {
        let peak: Float = 32
        let threshold = peak * PerthConfig.magmaskP
        let nFreq = 2
        let t = 2
        var mag = [Float](repeating: 0, count: nFreq * t)
        mag[0 * t + 0] = peak
        mag[0 * t + 1] = threshold + 1

        let m = Spectral.magmaskMargin(mag, frames: t, nFreq: nFreq)
        #expect(abs(m - 1) < 1e-5, "closest frame is 1.0 above the threshold, got \(m)")
    }
}
