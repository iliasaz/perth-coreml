import Foundation
import Testing

@testable import PerthCoreML

/// Golden vectors dumped by `converter/gen_test_fixtures.py` straight out of torch, scipy and
/// librosa -- not out of an earlier run of this Swift code, which would only pin the bugs in place.
///
/// Float arrays travel as base64 little-endian fp32, the same encoding `PerthAssets` uses, so the
/// tests parse them without a JSON-number round trip and without a dependency.
enum Fixture {
    struct Missing: Error, CustomStringConvertible {
        let name: String
        var description: String {
            "Fixtures/\(name).json is missing -- regenerate with "
                + "`.venv/bin/python converter/gen_test_fixtures.py`"
        }
    }

    static func load<T: Decodable>(_ name: String) throws -> T {
        guard let root = Bundle.module.resourceURL else { throw Missing(name: name) }
        let url = root.appending(path: "Fixtures/\(name).json")
        guard let data = try? Data(contentsOf: url) else { throw Missing(name: name) }
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// `Data(base64Encoded:)` gives no alignment guarantee, so copy rather than rebind.
    static func decode<T>(_ s: String, as _: T.Type = T.self) -> [T] {
        guard let d = Data(base64Encoded: s) else { return [] }
        let n = d.count / MemoryLayout<T>.size
        return [T](unsafeUninitializedCapacity: n) { buf, count in
            _ = d.copyBytes(to: UnsafeMutableRawBufferPointer(buf))
            count = n
        }
    }

    static func f32(_ s: String) -> [Float] { decode(s) }
    static func i32(_ s: String) -> [Int32] { decode(s) }
}

// MARK: - Metrics

/// `10*log10(sum(ref^2) / sum((ref-got)^2))`, accumulated in Double. `.infinity` when identical.
func snr(_ ref: [Float], _ got: [Float]) -> Double {
    precondition(ref.count == got.count)
    var signal = 0.0
    var noise = 0.0
    for i in ref.indices {
        signal += Double(ref[i]) * Double(ref[i])
        let d = Double(ref[i]) - Double(got[i])
        noise += d * d
    }
    if noise == 0 { return .infinity }
    return 10 * log10(signal / noise)
}

func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    return zip(a, b).reduce(Float(0)) { Swift.max($0, abs($1.0 - $1.1)) }
}

/// Index of the first element that differs, and by how much. `nil` when they are equal.
func firstMismatch(_ a: [Float], _ b: [Float]) -> (index: Int, got: Float, want: Float)? {
    precondition(a.count == b.count)
    for i in a.indices where a[i] != b[i] { return (i, a[i], b[i]) }
    return nil
}

// MARK: - Deterministic signals

/// SplitMix64. A test that draws from the system RNG is a test that fails once a month for
/// reasons nobody can reproduce.
struct Rng {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    mutating func uniform() -> Float {
        Float(next() >> 40) / Float(1 << 23) * 2 - 1
    }
}

func whiteNoise(_ n: Int, seed: UInt64, amplitude: Float = 0.5) -> [Float] {
    var rng = Rng(seed: seed)
    return (0..<n).map { _ in amplitude * rng.uniform() }
}

/// A voiced-speech-like signal: a decaying harmonic stack under a slow amplitude envelope.
///
/// The detector was trained on speech, and it is the thing being asserted on -- feeding it white
/// noise tests the tiling and the resampler but says nothing about whether the watermark took.
func harmonicSignal(_ n: Int, sampleRate: Int, f0: Float = 120, seed: UInt64 = 3) -> [Float] {
    var rng = Rng(seed: seed)
    let phases = (1...30).map { _ in rng.uniform() * .pi }
    var out = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let t = Float(i) / Float(sampleRate)
        var s: Float = 0
        for (k, phi) in phases.enumerated() {
            let h = Float(k + 1)
            s += sinf(2 * .pi * f0 * h * t + phi) / h
        }
        let env = 0.6 + 0.4 * sinf(2 * .pi * 2.5 * t)      // 2.5 Hz syllabic rate
        out[i] = 0.3 * env * s + 0.01 * rng.uniform()
    }
    return out
}
