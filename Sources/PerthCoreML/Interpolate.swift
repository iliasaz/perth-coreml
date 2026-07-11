import Foundation

/// `F.interpolate` along the time axis, reproducing ATen bit-for-bit.
///
/// The index arithmetic MUST be done in Float32. Doing it in Double is not "more accurate", it is
/// *wrong*: it lands on a different source index. Measured against torch 2.8.0 --
///   - nearest: Double index math mismatches on 571/1668 (T,S) pairs (34%). Example: `T=30, S=22,
///     d=11` -> fp32 gives 15, fp64 gives 14, torch gives 15.
///   - linear: Double gives `max|delta| = 3.05e-5`, an actual off-by-one.
///
/// Likewise the linear blend must use an explicit `fma`. Swift does not auto-contract
/// `a*b + c*d`, and the non-contracted form disagrees with torch in over half of all samples.
enum Interpolate {
    /// `mode: .linear, alignCorners: true`, applied per channel.
    ///
    /// `x` is `(channels, tIn)` channel-major. Returns `(channels, tOut)`.
    static func linear(_ x: [Float], channels: Int, tIn: Int, tOut: Int) -> [Float] {
        if tOut == tIn { return x }
        var out = [Float](repeating: 0, count: channels * tOut)
        // align_corners = true. ATen's `area_pixel_compute_scale` returns 0 when the output has
        // a single element, rather than dividing by zero.
        let scale: Float = tOut > 1 ? Float(tIn - 1) / Float(tOut - 1) : 0
        for d in 0..<tOut {
            let r: Float = scale * Float(d)
            let i0 = min(Int(r.rounded(.down)), tIn - 1)
            let i1 = i0 < tIn - 1 ? i0 + 1 : i0
            let lam: Float = r - Float(i0)
            let inv: Float = 1 - lam
            for c in 0..<channels {
                let a = x[c * tIn + i0]
                let b = x[c * tIn + i1]
                out[c * tOut + d] = (b * lam).addingProduct(a, inv)   // fmaf(a, inv, b*lam)
            }
        }
        return out
    }

    /// `mode: .nearest`, single channel. PyTorch's "nearest" actually floors, asymmetrically.
    static func nearest(_ x: [Float], tIn: Int, tOut: Int) -> [Float] {
        if tOut == tIn { return x }
        let scale: Float = Float(tIn) / Float(tOut)         // Float32, NOT Double
        return (0..<tOut).map { d in
            x[min(Int((Float(d) * scale).rounded(.down)), tIn - 1)]
        }
    }
}
