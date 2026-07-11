import Foundation
import Testing
import simd

@testable import PerthCoreML

/// The two scalar operations at the very end of `getWatermark`, where the obvious Swift spelling is
/// the wrong one. Both are one line of code and both silently change the detector's verdict.
@Suite("Score rounding and clipping")
struct NumericsTests {

    /// `torch.round` is round-half-to-EVEN. The clip before it saturates scores to exactly 0.0 and
    /// 1.0 often enough that the tie case is not hypothetical -- a raw watermarked score of 1.00034
    /// clips to exactly 1.0.
    @Test("the score rounds half-to-even, as torch.round does")
    func bankersRounding() {
        #expect(Float(0.5).rounded(.toNearestOrEven) == 0)
        #expect(Float(1.5).rounded(.toNearestOrEven) == 2)
        #expect(Float(2.5).rounded(.toNearestOrEven) == 2)
        #expect(Float(0.5).nextUp.rounded(.toNearestOrEven) == 1)
        #expect(Float(0.5).nextDown.rounded(.toNearestOrEven) == 0)

        // The trap, spelled out: the default rounding is half-away-from-zero, so a score of
        // exactly 0.5 would be reported as watermarked.
        #expect(Float(0.5).rounded() == 1)
        #expect(Float(0.5).rounded() != Float(0.5).rounded(.toNearestOrEven))
    }

    /// A NaN score means "digital silence" (magmask is all-zero, so the masked mean divides by
    /// zero). Stock Perth returns NaN there. A clamp that swallows NaN reports 0.0 instead --
    /// "not watermarked" -- which is a wrong answer rather than an absent one.
    @Test("the clip propagates NaN, as torch.clip does")
    func nanClip() {
        let nan = Float.nan
        #expect(Swift.min(Swift.max(nan, 0), 1).isNaN)

        // The traps, spelled out: both of these turn a NaN score into a confident 0.0.
        #expect(!Float.minimum(Float.maximum(nan, 0), 1).isNaN)
        #expect(Float.minimum(Float.maximum(nan, 0), 1) == 0)
        #expect(!simd_clamp(nan, 0, 1).isNaN)
        #expect(simd_clamp(nan, 0, 1) == 0)

        // And the clip still has to clip: the raw watermarked score really does exceed 1.
        #expect(Swift.min(Swift.max(Float(1.00034), 0), 1) == 1)
        #expect(Swift.min(Swift.max(Float(-0.2), 0), 1) == 0)
    }

    @Test("NaN survives the round")
    func nanRound() {
        #expect(Float.nan.rounded(.toNearestOrEven).isNaN)
    }
}
