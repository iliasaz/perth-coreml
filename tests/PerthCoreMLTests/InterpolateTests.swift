import Foundation
import Testing

@testable import PerthCoreML

struct InterpFixture: Decodable {
    struct LinearCase: Decodable {
        let channels: Int
        let tIn: Int
        let tOut: Int
        let x: String
        let y: String
    }
    struct NearestCase: Decodable {
        let tIn: Int
        let tOut: Int
        let x: String
        let y: String
        let idx: String
    }
    struct Probe: Decodable {
        let tIn: Int
        let tOut: Int
        let d: Int
        let sourceIndex: Int
    }
    let torch: String
    let linear: [LinearCase]
    let nearest: [NearestCase]
    let fp32IndexProbe: Probe
}

@Suite("Interpolate (torch-exact)")
struct InterpolateTests {

    @Test("linear align_corners=true is bit-exact against F.interpolate")
    func linearMatchesTorch() throws {
        let fx: InterpFixture = try Fixture.load("interp")
        #expect(fx.torch.hasPrefix("2.8"), "fixtures were dumped from torch \(fx.torch)")
        #expect(fx.linear.count > 50)

        var mismatched = 0
        var worst: Float = 0
        for c in fx.linear {
            let got = Interpolate.linear(Fixture.f32(c.x), channels: c.channels,
                                         tIn: c.tIn, tOut: c.tOut)
            let want = Fixture.f32(c.y)
            #expect(got.count == want.count, "C=\(c.channels) \(c.tIn)->\(c.tOut)")
            guard got.count == want.count else { continue }
            if let m = firstMismatch(got, want) {
                mismatched += 1
                worst = Swift.max(worst, maxAbsDiff(got, want))
                Issue.record("""
                    C=\(c.channels) \(c.tIn)->\(c.tOut) at \(m.index): \
                    got \(m.got), torch \(m.want)
                    """)
            }
        }
        #expect(mismatched == 0, "\(mismatched)/\(fx.linear.count) cases, max|d| = \(worst)")
    }

    @Test("nearest is bit-exact against F.interpolate, and picks torch's source index")
    func nearestMatchesTorch() throws {
        let fx: InterpFixture = try Fixture.load("interp")
        #expect(fx.nearest.count > 30)

        for c in fx.nearest {
            let got = Interpolate.nearest(Fixture.f32(c.x), tIn: c.tIn, tOut: c.tOut)
            let want = Fixture.f32(c.y)
            #expect(got.count == want.count, "\(c.tIn)->\(c.tOut)")
            guard got.count == want.count else { continue }
            if let m = firstMismatch(got, want) {
                Issue.record("\(c.tIn)->\(c.tOut) at \(m.index): got \(m.got), torch \(m.want)")
            }

            // A ramp makes the chosen source index observable directly, so a failure says which
            // index the gather landed on rather than just "the numbers differ".
            let ramp = (0..<c.tIn).map(Float.init)
            let idx = Interpolate.nearest(ramp, tIn: c.tIn, tOut: c.tOut).map { Int32($0) }
            #expect(idx == Fixture.i32(c.idx), "\(c.tIn)->\(c.tOut) index selection")
        }
    }

    /// The index arithmetic must be done in Float32. fp64 is not "more accurate" here, it lands on
    /// a different source sample -- 34 % of (T,S) pairs disagree.
    @Test("nearest index math is fp32, not fp64")
    func fp32IndexMath() throws {
        let fx: InterpFixture = try Fixture.load("interp")
        let p = fx.fp32IndexProbe
        #expect((p.tIn, p.tOut, p.d) == (30, 22, 11))

        let ramp = (0..<p.tIn).map(Float.init)
        let got = Interpolate.nearest(ramp, tIn: p.tIn, tOut: p.tOut)
        #expect(Int(got[p.d]) == p.sourceIndex, "torch selects source index \(p.sourceIndex)")
        #expect(p.sourceIndex == 15)

        // The trap, spelled out: the same expression in Double picks 14.
        let f64 = Int((Double(p.d) * (Double(p.tIn) / Double(p.tOut))).rounded(.down))
        #expect(f64 == 14, "the fp64 form is expected to be wrong here; it gave \(f64)")
        #expect(f64 != p.sourceIndex)
    }

    @Test("tOut == tIn is the identity")
    func identity() {
        let x = whiteNoise(37 * 3, seed: 9)
        #expect(Interpolate.linear(x, channels: 3, tIn: 37, tOut: 37) == x)
        let g = whiteNoise(37, seed: 10)
        #expect(Interpolate.nearest(g, tIn: 37, tOut: 37) == g)
    }

    @Test("the detect path's branch lengths truncate toward zero")
    func branchLengths() {
        // Python: int(t * 1.25) and int(t * 0.75). Integer division reproduces that only because
        // t is non-negative.
        for t in [1, 2, 3, 7, 30, 101, 301, 1001] {
            #expect(5 * t / 4 == Int(Double(t) * 1.25), "t=\(t)")
            #expect(3 * t / 4 == Int(Double(t) * 0.75), "t=\(t)")
        }
    }
}
