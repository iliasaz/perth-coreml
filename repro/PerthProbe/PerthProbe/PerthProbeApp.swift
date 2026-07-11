import CoreML
import PerthCoreML
import SwiftUI

/// On-device gate for the two success criteria that a Mac cannot settle:
///   1. does the model actually run on the iPhone's ANE, and
///   2. does it still agree with Python there? (iPhone ANE numerics are NOT Mac ANE numerics --
///      a sibling project measured 0.9999 on Mac and 0.9828 on device for the same graph.)
///
/// Everything goes through `print` + `fflush`, because `os.Logger` output never reaches
/// `--console`. One (precision x compute-unit) pair per process when `PROBE_ONLY` is set: a
/// failed `MLModel` load SIGBUSes inside CoreML's AOT compiler and takes the whole app with it,
/// so a single bad combination would otherwise destroy every result in the launch.
@main
struct PerthProbeApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var lines: [String] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, l in
                    Text(l).font(.system(size: 10, design: .monospaced))
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }
        .task { Probe.run { l in lines.append(l) } }
    }
}

// MARK: -

enum Probe {
    static func say(_ s: String, _ sink: (String) -> Void) {
        print(s)
        fflush(stdout)
        sink(s)
    }

    /// Hand-rolled, because `String(format:)` with `%s` and a Swift String is undefined behaviour
    /// (it wants a C string) and kills the app mid-probe -- which reads exactly like a CoreML load
    /// crash and sends you hunting in the wrong place.
    static func pad(_ s: String, _ n: Int) -> String {
        s.count >= n ? s + " " : s + String(repeating: " ", count: n - s.count)
    }

    static func dec(_ v: Double, _ places: Int) -> String {
        if v.isNaN { return "nan" }
        if v.isInfinite { return v > 0 ? "inf" : "-inf" }
        let m = pow(10.0, Double(places))
        let r = (v * m).rounded() / m
        var s = "\(r)"
        if places == 0, let dot = s.firstIndex(of: ".") { s = String(s[s.startIndex..<dot]) }
        return s
    }

    static func run(_ sink: @escaping (String) -> Void) {
        let env = ProcessInfo.processInfo.environment
        let only = env["PROBE_ONLY"]                      // e.g. "fp16:ane"
        let rates = (env["PROBE_RATES"] ?? "32000,24000").split(separator: ",").map { Int($0)! }

        guard let dir = Bundle.main.resourceURL else {
            say("PROBE: no resource URL", sink); return
        }

        let fx: Fixtures
        do { fx = try Fixtures(dir: dir) } catch {
            say("PROBE: fixtures failed: \(error)", sink); return
        }
        say("PROBE: start  models=\(dir.lastPathComponent)", sink)

        let configs: [(String, Bool, String, MLComputeUnits)] = [
            ("fp32:cpu", true, "cpuOnly", .cpuOnly),
            ("fp16:cpu", false, "cpuOnly", .cpuOnly),
            ("fp16:gpu", false, "cpuAndGPU", .cpuAndGPU),
            ("fp16:ane", false, "cpuAndNeuralEngine", .cpuAndNeuralEngine),
            ("fp16:all", false, "all", .all),
        ]

        for sr in rates {
            let (input, expected, wantConf) = fx.forRate(sr)
            say("", sink)
            say("=== \(sr) Hz   \(input.count) samples in, python -> \(expected.count) out, "
                + "python detect(wm)=\(dec(Double(wantConf), 6))", sink)
            say(pad("cfg", 9) + pad("load", 8) + pad("apply", 9) + pad("cos", 11)
                + pad("SNR dB", 9) + pad("detect", 10) + "verdict", sink)

            for (tag, fp32, _, cu) in configs {
                if let only, only != tag { continue }
                autoreleasepool {
                    do {
                        let t0 = Date()
                        let perth = try PerthWatermarker(modelDirectory: dir, computeUnits: cu,
                                                         useFP32: fp32,
                                                         lengthPolicy: .pythonParity)
                        let load = -t0.timeIntervalSinceNow

                        let t1 = Date()
                        let got = try perth.applyWatermark(input, sampleRate: sr)
                        let apply = -t1.timeIntervalSinceNow

                        let conf = try perth.getWatermark(got, sampleRate: sr, round: false)

                        let c = cosine(got, expected)
                        let s = snr(expected, got)
                        let lenOK = got.count == expected.count
                        let ok = lenOK && c >= 0.9999 && conf.rounded() == 1

                        say(pad(tag, 9) + pad(dec(load, 2) + "s", 8)
                            + pad(dec(apply * 1000, 0) + "ms", 9) + pad(dec(c, 7), 11)
                            + pad(dec(s, 1), 9) + pad(dec(Double(conf), 6), 10)
                            + (ok ? "PASS" : "FAIL"), sink)
                        if !lenOK {
                            say("          length \(got.count) != \(expected.count)", sink)
                        }
                    } catch {
                        say("\(tag)  ERROR: \(error)", sink)
                    }
                }
            }
        }
        say("", sink)
        say("PROBE: done", sink)

        // A SwiftUI app never terminates on its own, so `devicectl --console` would block until
        // its timeout and we'd never see the exit. Quit unless someone wants to read the screen.
        if ProcessInfo.processInfo.environment["PROBE_STAY"] == nil {
            exit(0)
        }
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<n {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        let d = (na * nb).squareRoot()
        return d == 0 ? 0 : dot / d
    }

    static func snr(_ ref: [Float], _ got: [Float]) -> Double {
        let n = min(ref.count, got.count)
        var sig = 0.0, err = 0.0
        for i in 0..<n {
            sig += Double(ref[i]) * Double(ref[i])
            let e = Double(ref[i]) - Double(got[i])
            err += e * e
        }
        return err == 0 ? .infinity : 10 * log10(sig / err)
    }
}

// MARK: -

struct Fixtures {
    let input32: [Float], expected32: [Float], conf32: Float
    let input24: [Float], expected24: [Float], conf24: Float

    init(dir: URL) throws {
        func f32(_ name: String) throws -> [Float] {
            let d = try Data(contentsOf: dir.appendingPathComponent(name))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        input32 = try f32("input32.f32")
        expected32 = try f32("expected32.f32")
        input24 = try f32("input24.f32")
        expected24 = try f32("expected24.f32")

        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: dir.appendingPathComponent("meta.json"))) as! [String: Any]
        conf32 = Float((meta["sr32"] as! [String: Any])["detect_watermarked"] as! Double)
        conf24 = Float((meta["sr24"] as! [String: Any])["detect_watermarked"] as! Double)
    }

    func forRate(_ sr: Int) -> ([Float], [Float], Float) {
        sr == 32000 ? (input32, expected32, conf32) : (input24, expected24, conf24)
    }
}
