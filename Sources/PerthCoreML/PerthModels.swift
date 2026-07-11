import CoreML
import Foundation

/// The two CoreML conv stacks, plus the compiled-model cache.
///
/// Only the convolutions live here. Everything else -- STFT/ISTFT, dB (de)normalisation, magmask,
/// interpolation, masked-mean, softmax, the residual add -- stays on the host in fp32, because
/// CoreML has no complex-number support and those ops are free next to the conv stacks anyway.
final class PerthModels {
    private let encoder: MLModel
    private let decoder: MLModel
    let window: Int

    /// - Parameters:
    ///   - directory: holds `PerthEncoder.mlpackage` and `PerthDecoder.mlpackage` (or the
    ///     `_fp32` variants).
    ///   - computeUnits: `.cpuAndNeuralEngine` or `.all` for the ANE tier; `.cpuOnly` with the
    ///     fp32 packages for the numeric-parity reference tier.
    ///   - useFP32: load the fp32 reference packages instead of the fp16 ones.
    init(directory: URL,
         computeUnits: MLComputeUnits = .all,
         useFP32: Bool = false,
         window: Int = PerthConfig.modelWindow) throws {
        self.window = window
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let suffix = useFP32 ? "_fp32" : ""
        encoder = try PerthModels.load("PerthEncoder\(suffix)", in: directory, config: cfg)
        decoder = try PerthModels.load("PerthDecoder\(suffix)", in: directory, config: cfg)
    }

    /// Loads a package, reusing a persisted `.mlmodelc` at a STABLE path.
    ///
    /// `MLModel.compileModel(at:)` returns a fresh `tmp/<UUID>.mlmodelc` on every call, and the
    /// ANE's AOT cache is keyed on the compiled model's identity -- so a fresh path each launch
    /// means a cache miss and a full ANE recompile every single launch. Persisting the compiled
    /// model at a stable path is what makes warm loads warm.
    private static func load(_ name: String, in dir: URL,
                             config: MLModelConfiguration) throws -> MLModel {
        let pkg = dir.appendingPathComponent("\(name).mlpackage")
        guard FileManager.default.fileExists(atPath: pkg.path) else {
            throw PerthError.modelNotFound(name: name, directory: dir)
        }
        if let cached = try? CompiledModelCache.compiledURL(for: pkg, name: name),
           let model = try? MLModel(contentsOf: cached, configuration: config) {
            return model
        }
        // Never fail to load just because the cache misbehaved.
        let tmp = try MLModel.compileModel(at: pkg)
        return try MLModel(contentsOf: tmp, configuration: config)
    }

    // MARK: - Predictions

    /// (subband, validity mask) -> watermark residual, both `(1, 128, W)`.
    func encode(x: MLMultiArray, mask: MLMultiArray) throws -> MLMultiArray {
        let out = try encoder.prediction(from: try MLDictionaryFeatureProvider(
            dictionary: ["sub_mag": x, "mask": mask]))
        guard let r = out.featureValue(for: "residual")?.multiArrayValue else {
            throw PerthError.badModelOutput("residual")
        }
        return r
    }

    /// The three decoder branches in one predict. Each output is `(1, 2, W)`: channel 0 is the
    /// branch's attention logit, channel 1 its watermark estimate.
    ///
    /// A branch that has run out of tiles is fed `mask = 0`, which contributes nothing to either
    /// side of the masked mean -- so exhausted branches cost nothing and need no special case.
    func decode(slow: (MLMultiArray, MLMultiArray),
                norm: (MLMultiArray, MLMultiArray),
                fast: (MLMultiArray, MLMultiArray)) throws
    -> (slow: MLMultiArray, norm: MLMultiArray, fast: MLMultiArray) {
        let out = try decoder.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            "slow_x": slow.0, "slow_m": slow.1,
            "norm_x": norm.0, "norm_m": norm.1,
            "fast_x": fast.0, "fast_m": fast.1,
        ]))
        guard let s = out.featureValue(for: "slow_out")?.multiArrayValue,
              let n = out.featureValue(for: "norm_out")?.multiArrayValue,
              let f = out.featureValue(for: "fast_out")?.multiArrayValue else {
            throw PerthError.badModelOutput("slow_out/norm_out/fast_out")
        }
        return (s, n, f)
    }
}

/// Persists compiled `.mlmodelc` directories at a stable, content-stamped path.
enum CompiledModelCache {
    static func compiledURL(for package: URL, name: String) throws -> URL {
        let root = try FileManager.default.url(for: .applicationSupportDirectory,
                                               in: .userDomainMask,
                                               appropriateFor: nil, create: true)
            .appendingPathComponent("com.perth.coreml/CompiledModels", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let key = "\(name)-\(stamp(of: package))"
        var dst = root.appendingPathComponent("\(key).mlmodelc", isDirectory: true)

        // Warm hit only if the compile actually finished: a partially written .mlmodelc would
        // otherwise be reused forever.
        if FileManager.default.fileExists(
            atPath: dst.appendingPathComponent("coremldata.bin").path) {
            return dst
        }

        let compiled = try MLModel.compileModel(at: package)
        try? FileManager.default.removeItem(at: dst)
        try FileManager.default.moveItem(at: compiled, to: dst)

        // Regenerable derived data -- don't back it up.
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? dst.setResourceValues(values)
        return dst
    }

    /// FNV-1a over each file's relative path, size and mtime. Not `Hasher` -- that is seeded
    /// per-process, so it would never produce the same key twice.
    private static func stamp(of package: URL) -> String {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ s: String) {
            for b in s.utf8 { h = (h ^ UInt64(b)) &* 0x100_0000_01b3 }
        }
        let fm = FileManager.default
        let files = (fm.enumerator(at: package, includingPropertiesForKeys: [.fileSizeKey,
                                                                             .contentModificationDateKey])?
            .compactMap { $0 as? URL } ?? []).sorted { $0.path < $1.path }
        for f in files {
            let v = try? f.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            mix(f.lastPathComponent)
            mix(String(v?.fileSize ?? 0))
            mix(String(Int((v?.contentModificationDate ?? .distantPast).timeIntervalSince1970 * 1000)))
        }
        return String(h, radix: 16)
    }
}
