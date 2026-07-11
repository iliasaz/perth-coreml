import Foundation

/// One static-width window of the conv stack, plus which of its outputs are trustworthy.
struct Tile {
    /// Where this window starts reading the input.
    let inStart: Int
    /// How many real (non-padded) frames it holds. The rest of the window is zeros and the
    /// validity mask is 0 there.
    let realFrames: Int
    /// First output index within the window that this tile owns.
    let keepLo: Int
    /// How many outputs it owns, starting at `keepLo`.
    let keepCount: Int

    /// Where those outputs land in the full-length result.
    var outStart: Int { inStart + keepLo }
}

/// Windows covering `t` frames at the CoreML model's static width.
///
/// An interior window has a fake cut at each edge, so its first and last `halo` outputs are wrong
/// and get discarded -- hence the stride of `w - 2*halo`. The first window's left edge and the
/// final window's right edge are *true* signal boundaries, where the model's own padding (plus the
/// validity mask) is already correct, so nothing is discarded there.
///
/// Coverage is gap-free and overlap-free by construction: the next window's first owned output is
/// `inStart + halo`, which is exactly where this one stops.
///
/// `t <= w` collapses to a single window with `keepLo == 0` -- there is no short-signal special
/// case, which is the whole reason the validity mask exists.
func tilePlan(frames t: Int,
              window w: Int = PerthConfig.modelWindow,
              halo h: Int = PerthConfig.halo) -> [Tile] {
    precondition(w > 2 * h, "window \(w) too small for halo \(h)")
    guard t > 0 else { return [] }
    let stride = w - 2 * h
    var tiles: [Tile] = []
    var outStart = 0
    while outStart < t {
        let inStart = outStart == 0 ? 0 : outStart - h
        let keepLo = outStart - inStart                 // 0 on the first window, halo after
        let realFrames = min(w, t - inStart)
        let owned = min(outStart == 0 ? w - h : stride, t - outStart, realFrames - keepLo)
        tiles.append(Tile(inStart: inStart, realFrames: realFrames,
                          keepLo: keepLo, keepCount: owned))
        outStart += owned
    }
    return tiles
}
