import CoreML
import Foundation

/// How to reconcile Perth's lossy output length with the caller's.
public enum LengthPolicy: Sendable {
    /// Reproduce Python exactly, including the truncation. `apply` returns up to `hop - 1` fewer
    /// samples than it was given (up to 239 at 24 kHz, ~10 ms), because Perth's ISTFT is called
    /// without `length:` and drops the trailing partial frame.
    ///
    /// Required for any comparison against Python -- a cosine gate is meaningless if the two
    /// signals are different lengths.
    case pythonParity
    /// Return exactly as many samples as the caller passed in. Pads the 32 kHz signal up to a
    /// whole number of frames before the STFT, then trims back. The recovered tail really is
    /// watermarked, and Python's detector still fires on the result.
    case preserveLength
}

/// Perth-Net Implicit: embeds an imperceptible watermark in a magnitude spectrogram, and detects
/// one.
///
/// Mirrors `perth.PerthImplicitWatermarker`. The conv stacks run in CoreML (ANE-resident at a
/// static window); everything else runs on the host in fp32.
public final class PerthWatermarker {
    private let models: PerthModels
    private let stft: STFT
    private let lengthPolicy: LengthPolicy

    public init(modelDirectory: URL,
                computeUnits: MLComputeUnits = .all,
                useFP32: Bool = false,
                lengthPolicy: LengthPolicy = .preserveLength) throws {
        self.models = try PerthModels(directory: modelDirectory,
                                      computeUnits: computeUnits, useFP32: useFP32)
        self.stft = STFT(window: PerthAssets.stftWindow)
        self.lengthPolicy = lengthPolicy
    }

    // MARK: - Embed

    /// Watermarks `wav`. Resamples to Perth's native 32 kHz and back if needed.
    ///
    /// Signals shorter than ~43 ms (1025 samples at 32 kHz) cannot be watermarked -- the STFT
    /// reflect-pads by 1024 -- and are returned unchanged rather than throwing, so a caller
    /// stitching short chunks never crashes.
    public func applyWatermark(_ wav: [Float], sampleRate: Int) throws -> [Float] {
        let resample = sampleRate != PerthConfig.sampleRate
        if resample && sampleRate != 24_000 { throw PerthError.unsupportedSampleRate(sampleRate) }

        let x32 = resample ? Resampler.up24to32Apply(wav) : wav
        guard x32.count >= PerthConfig.minSamples32k else { return wav }

        let watermarked32 = try embed32k(x32)
        guard resample else { return watermarked32 }

        var out = Resampler.down32to24Apply(watermarked32)
        if lengthPolicy == .preserveLength { out = fit(out, to: wav.count) }
        return out
    }

    /// The 32 kHz core: STFT -> encode residual -> gate -> add -> ISTFT.
    private func embed32k(_ signal: [Float]) throws -> [Float] {
        // .preserveLength: pad up to a whole frame so the ISTFT hands back everything. The extra
        // frame changes the overlap-add only near the seam.
        var x = signal
        if lengthPolicy == .preserveLength {
            let rem = x.count % PerthConfig.hopSize
            if rem != 0 { x += [Float](repeating: 0, count: PerthConfig.hopSize - rem + PerthConfig.hopSize) }
        }

        let (mag, phase, t) = try stft.forward(x)
        let gate = Spectral.magmask(mag, frames: t)          // needs all 1025 bins => host-side

        // The residual is produced only for the low 128 bins.
        let sub = Array(mag[0..<(PerthConfig.subband * t)])
        let residual = try encodeTiled(sub, frames: t)

        var wm = mag
        for k in 0..<PerthConfig.subband {
            for f in 0..<t {
                wm[k * t + f] += residual[k * t + f] * gate[f]
            }
        }
        var out = stft.inverse(mag: wm, phase: phase, frames: t)
        if lengthPolicy == .preserveLength { out = fit(out, to: signal.count) }
        return out
    }

    /// Runs the encoder conv stack over `t` frames using fixed-width windows.
    private func encodeTiled(_ sub: [Float], frames t: Int) throws -> [Float] {
        let w = models.window
        var residual = [Float](repeating: 0, count: PerthConfig.subband * t)
        for tile in tilePlan(frames: t, window: w) {
            let (x, m) = try window(sub, channels: PerthConfig.subband, frames: t, tile: tile, w: w)
            let out = try models.encode(x: x, mask: m)
            let p = out.dataPointer.bindMemory(to: Float.self, capacity: PerthConfig.subband * w)
            for c in 0..<PerthConfig.subband {
                for i in 0..<tile.keepCount {
                    residual[c * t + tile.outStart + i] = p[c * w + tile.keepLo + i]
                }
            }
        }
        return residual
    }

    // MARK: - Detect

    /// The watermark confidence for `wav`: 1 if watermarked, 0 if not (or the raw score).
    ///
    /// Returns NaN for digitally silent audio -- `magmask` is then all-zero and the masked mean
    /// divides by zero. That is what stock Perth does, and we reproduce it rather than inventing
    /// a value.
    public func getWatermark(_ wav: [Float], sampleRate: Int, round: Bool = true) throws -> Float {
        let resample = sampleRate != PerthConfig.sampleRate
        if resample && sampleRate != 24_000 { throw PerthError.unsupportedSampleRate(sampleRate) }
        // Python's detect path resamples with res_type: "polyphase", NOT the default soxr.
        let x32 = resample ? Resampler.up24to32Detect(wav) : wav
        guard x32.count >= PerthConfig.minSamples32k else {
            throw PerthError.signalTooShort(samples: x32.count, minimum: PerthConfig.minSamples32k)
        }

        let (mag, _, t) = try stft.forward(x32)
        let gate = Spectral.magmask(mag, frames: t)
        let sub = Array(mag[0..<(PerthConfig.subband * t)])

        // Branch lengths: Python's int(T*1.25) / int(T*0.75) truncate toward zero.
        let lens = [5 * t / 4, t, 3 * t / 4]
        let xs = lens.map { Interpolate.linear(sub, channels: PerthConfig.subband,
                                               tIn: t, tOut: $0) }
        let gates = lens.map { Interpolate.nearest(gate, tIn: t, tOut: $0) }
        let plans = lens.map { tilePlan(frames: $0, window: models.window) }
        let nTiles = plans.map(\.count).max() ?? 0

        // The masked mean is linear, so its numerator and denominator accumulate across tiles.
        var numAttn = [Double](repeating: 0, count: 3)
        var numWmark = [Double](repeating: 0, count: 3)
        var den = [Double](repeating: 0, count: 3)
        let w = models.window

        for i in 0..<nTiles {
            var feeds: [(MLMultiArray, MLMultiArray)] = []
            for b in 0..<3 {
                if i < plans[b].count {
                    feeds.append(try window(xs[b], channels: PerthConfig.subband,
                                            frames: lens[b], tile: plans[b][i], w: w))
                } else {
                    // Exhausted branch: an all-zero mask contributes nothing to either sum.
                    feeds.append((try zeros([1, NSNumber(value: PerthConfig.subband),
                                             NSNumber(value: w)]),
                                  try zeros([1, 1, NSNumber(value: w)])))
                }
            }
            let out = try models.decode(slow: feeds[0], norm: feeds[1], fast: feeds[2])
            for (b, arr) in [out.slow, out.norm, out.fast].enumerated() {
                guard i < plans[b].count else { continue }
                let tile = plans[b][i]
                let p = arr.dataPointer.bindMemory(to: Float.self, capacity: 2 * w)
                for j in 0..<tile.keepCount {
                    let g = Double(gates[b][tile.outStart + j])
                    numAttn[b] += Double(p[0 * w + tile.keepLo + j]) * g
                    numWmark[b] += Double(p[1 * w + tile.keepLo + j]) * g
                    den[b] += g
                }
            }
        }

        let attn = (0..<3).map { Float(numAttn[$0] / den[$0]) }     // den == 0 => NaN, as in Python
        let wmks = (0..<3).map { Float(numWmark[$0] / den[$0]) }

        let m = max(attn[0], max(attn[1], attn[2]))
        let e = attn.map { expf($0 - m) }
        let sum = (e[0] + e[1]) + e[2]
        let score = ((wmks[0] * e[0] / sum) + (wmks[1] * e[1] / sum)) + (wmks[2] * e[2] / sum)

        // Swift.min/max propagate NaN the way torch.clip does. Float.minimum and simd_clamp do
        // NOT -- they turn NaN into 0.0, which would silently report "not watermarked".
        let clipped = Swift.min(Swift.max(score, 0), 1)
        // torch.round is round-half-to-EVEN. Float.rounded() is half-away-from-zero.
        return round ? clipped.rounded(.toNearestOrEven) : clipped
    }

    // MARK: - Helpers

    /// Slices one window out of a `(channels, frames)` array, zero-padded, plus its validity mask.
    ///
    /// The mask is 1 on real frames and 0 on padding -- INCLUDING silent real frames. It is not
    /// `magmask`. Confusing the two is the easiest bug in this project to write: the model masks
    /// activations after every layer, and feeding it `magmask` would change what the model
    /// computes rather than just neutralising the padding.
    private func window(_ x: [Float], channels: Int, frames t: Int,
                        tile: Tile, w: Int) throws -> (MLMultiArray, MLMultiArray) {
        let xa = try zeros([1, NSNumber(value: channels), NSNumber(value: w)])
        let ma = try zeros([1, 1, NSNumber(value: w)])
        let xp = xa.dataPointer.bindMemory(to: Float.self, capacity: channels * w)
        let mp = ma.dataPointer.bindMemory(to: Float.self, capacity: w)
        for c in 0..<channels {
            for i in 0..<tile.realFrames {
                xp[c * w + i] = x[c * t + tile.inStart + i]
            }
        }
        for i in 0..<tile.realFrames { mp[i] = 1 }
        return (xa, ma)
    }

    private func zeros(_ shape: [NSNumber]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape, dataType: .float32)
        memset(a.dataPointer, 0, a.count * MemoryLayout<Float>.size)
        return a
    }

    /// Pad with zeros or truncate so the result is exactly `n` samples.
    private func fit(_ x: [Float], to n: Int) -> [Float] {
        if x.count == n { return x }
        if x.count > n { return Array(x[0..<n]) }
        return x + [Float](repeating: 0, count: n - x.count)
    }
}
