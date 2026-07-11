import Foundation
import Testing

@testable import PerthCoreML

struct TilingFixture: Decodable {
    struct Case: Decodable {
        let w: Int
        let halo: Int
        let t: Int
        /// [inStart, keepLo, keepCount, realFrames], as Python's `tile_plan` yields them.
        let tiles: [[Int]]
    }
    let cases: [Case]
}

@Suite("Tiling")
struct TilingTests {

    /// Every frame is produced exactly once. A gap silently zeroes a slice of the residual; an
    /// overlap silently double-counts frames in the detector's masked mean.
    @Test("coverage is gap-free and overlap-free")
    func coverage() {
        let w = PerthConfig.modelWindow
        var ts = Set([1, 2, 3, 15, 16, 100, w - 2 * PerthConfig.halo, w - PerthConfig.halo,
                      w - 1, w, w + 1, w + 2, 2 * w, 2 * w + 1, 3 * w, 5000])
        ts.formUnion((0..<200).map { 1 + $0 * 37 })
        ts.formUnion([301, 376, 225, 1001, 1251, 750, 3001, 3751, 2250])

        for t in ts.sorted() {
            let tiles = tilePlan(frames: t, window: w)
            #expect(!tiles.isEmpty, "t=\(t)")

            var covered = [Bool](repeating: false, count: t)
            for tile in tiles {
                #expect(tile.keepCount > 0, "t=\(t): a tile that owns nothing never terminates")
                #expect(tile.inStart >= 0, "t=\(t)")
                #expect(tile.realFrames > 0 && tile.realFrames <= w, "t=\(t)")
                #expect(tile.inStart + tile.realFrames <= t, "t=\(t): reads past the signal")
                #expect(tile.keepLo >= 0, "t=\(t)")
                #expect(tile.keepLo + tile.keepCount <= tile.realFrames,
                        "t=\(t): keeps outputs the window never saw real input for")
                #expect(tile.outStart + tile.keepCount <= t, "t=\(t): writes past the output")

                for i in tile.outStart..<(tile.outStart + tile.keepCount) {
                    #expect(!covered[i], "t=\(t): frame \(i) is produced twice")
                    covered[i] = true
                }
            }
            #expect(covered.allSatisfy { $0 }, "t=\(t): \(covered.filter { !$0 }.count) frames unwritten")
        }
    }

    /// The first window keeps `w - halo` outputs and no more, even when it already holds the whole
    /// signal -- so one window suffices only up to `w - halo` frames, not up to `w`.
    @Test("t <= w - halo is a single window with nothing discarded")
    func shortSignal() {
        let w = PerthConfig.modelWindow
        let h = PerthConfig.halo
        for t in [1, 2, 15, 16, 500, w - h - 1, w - h] {
            let tiles = tilePlan(frames: t, window: w)
            #expect(tiles.count == 1, "t=\(t)")
            #expect(tiles[0].inStart == 0, "t=\(t)")
            #expect(tiles[0].keepLo == 0, "t=\(t): the first window's left edge is a real boundary")
            #expect(tiles[0].keepCount == t, "t=\(t)")
            #expect(tiles[0].realFrames == t, "t=\(t)")
        }
    }

    /// Between `w - halo` and `w` the signal fits in one window but the plan still cuts a second
    /// one for the last few frames. Wasteful, and the second window's outputs are recomputed from a
    /// truncated input -- but they are still inside the halo's dependency cone, so they are right.
    @Test("w - halo < t <= w still needs a second window")
    func fitsButStillSplits() {
        let w = PerthConfig.modelWindow
        let h = PerthConfig.halo
        for t in [w - h + 1, w - 1, w] {
            let tiles = tilePlan(frames: t, window: w)
            #expect(tiles.count == 2, "t=\(t)")
            #expect(tiles[0].keepCount == w - h, "t=\(t)")
            #expect(tiles[1].keepCount == t - (w - h), "t=\(t)")
            // Output j depends on input [j-halo, j+halo]; the second window starts halo frames
            // before the first output it owns, so every dependency it needs is inside it.
            #expect(tiles[1].inStart <= tiles[1].outStart - h, "t=\(t)")
        }
    }

    @Test("t == w + 1 needs a second window")
    func justOverWindow() {
        let w = PerthConfig.modelWindow
        let h = PerthConfig.halo
        let tiles = tilePlan(frames: w + 1, window: w)
        #expect(tiles.count == 2)
        #expect(tiles[0].keepCount == w - h)          // the last halo outputs sit at a fake cut
        #expect(tiles[1].inStart == w - h - h)
        #expect(tiles[1].keepLo == h)
        #expect(tiles[0].keepCount + tiles[1].keepCount == w + 1)
    }

    @Test("zero frames yields no tiles")
    func empty() {
        #expect(tilePlan(frames: 0, window: PerthConfig.modelWindow).isEmpty)
    }

    @Test("matches the Python tile_plan the converter was validated against")
    func matchesPython() throws {
        let fx: TilingFixture = try Fixture.load("tiling")
        #expect(!fx.cases.isEmpty)

        for c in fx.cases {
            let tiles = tilePlan(frames: c.t, window: c.w, halo: c.halo)
            #expect(tiles.count == c.tiles.count, "w=\(c.w) t=\(c.t)")
            guard tiles.count == c.tiles.count else { continue }
            for (i, tile) in tiles.enumerated() {
                let want = c.tiles[i]
                #expect([tile.inStart, tile.keepLo, tile.keepCount, tile.realFrames] == want,
                        "w=\(c.w) t=\(c.t) tile \(i)")
            }
        }
    }
}
