import AVFoundation
import CoreML
import Foundation
import PerthCoreML

@main
struct PerthCLI {
    static func main() throws {
        var args = Array(CommandLine.arguments.dropFirst())

        func flag(_ name: String) -> Bool {
            guard let i = args.firstIndex(of: name) else { return false }
            args.remove(at: i)
            return true
        }
        func option(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
            let v = args[i + 1]
            args.removeSubrange(i...(i + 1))
            return v
        }

        let detect = flag("--detect")
        let fp32 = flag("--fp32")
        let parity = flag("--python-parity")
        let cuName = option("--cu") ?? "all"
        guard let modelDir = option("--models") else { die("--models <dir> is required") }
        let outPath = option("--out")
        guard let inPath = args.first else {
            die("usage: perth-cli <in.wav> --models <dir> [--out <o.wav>] [--detect] [--fp32] "
                + "[--cu cpu|gpu|ane|all] [--python-parity]")
        }

        let cu: MLComputeUnits
        switch cuName {
        case "cpu": cu = .cpuOnly
        case "gpu": cu = .cpuAndGPU
        case "ane": cu = .cpuAndNeuralEngine
        case "all": cu = .all
        default: die("--cu must be cpu|gpu|ane|all")
        }

        let (samples, sr) = try readWav(inPath)
        let secs = Double(samples.count) / Double(sr)
        print("in : \(inPath)  \(samples.count) samples @ \(sr) Hz  (\(f(secs, 2))s)")

        let t0 = Date()
        let perth = try PerthWatermarker(
            modelDirectory: URL(fileURLWithPath: modelDir),
            computeUnits: cu, useFP32: fp32,
            lengthPolicy: parity ? .pythonParity : .preserveLength)
        print("load: \(f(-t0.timeIntervalSinceNow, 2))s  (\(fp32 ? "fp32" : "fp16"), \(cuName))")

        if detect {
            let t = Date()
            let raw = try perth.getWatermark(samples, sampleRate: sr, round: false)
            let rounded = try perth.getWatermark(samples, sampleRate: sr)
            print("detect: raw=\(raw)  rounded=\(rounded)  "
                  + "(\(f(-t.timeIntervalSinceNow * 1000, 0)) ms)")
            return
        }

        let t = Date()
        let wm = try perth.applyWatermark(samples, sampleRate: sr)
        let ms = -t.timeIntervalSinceNow * 1000
        print("apply: \(wm.count) samples  (\(f(ms, 0)) ms, \(f(secs * 1000 / ms, 1))x realtime)")

        // Embedding needs only PerthEncoder, so a decoder-less model dir is a legitimate way to
        // deploy. Don't turn the courtesy round-trip check into a hard requirement for it.
        do {
            print("verify: detect(watermarked) = "
                  + "\(try perth.getWatermark(wm, sampleRate: sr, round: false))")
        } catch let e as PerthError {
            print("verify: skipped (\(e))")
        }

        if let outPath {
            try writeWav(wm, sampleRate: sr, to: outPath)
            print("out: \(outPath)")
        }
    }

    // MARK: - IO

    static func readWav(_ path: String) throws -> ([Float], Int) {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: file.fileFormat.sampleRate,
                                channels: 1, interleaved: false)!

        // read(into:) is not guaranteed to fill the buffer in one call -- it returned 159724 of
        // 160000 frames on a plain float32 wav, which silently shortened the signal by most of a
        // hop and shifted the frame count. Loop until the file stops yielding frames.
        var samples: [Float] = []
        samples.reserveCapacity(Int(file.length))
        let chunk = AVAudioFrameCount(1 << 16)
        while file.framePosition < file.length {
            guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: chunk) else {
                die("could not allocate a read buffer for \(path)")
            }
            try file.read(into: buf)
            guard buf.frameLength > 0 else { break }
            samples.append(contentsOf: UnsafeBufferPointer(start: buf.floatChannelData![0],
                                                           count: Int(buf.frameLength)))
        }
        return (samples, Int(file.fileFormat.sampleRate))
    }

    static func writeWav(_ x: [Float], sampleRate: Int, to path: String) throws {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        let file = try AVAudioFile(forWriting: URL(fileURLWithPath: path), settings: fmt.settings)
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt,
                                         frameCapacity: AVAudioFrameCount(x.count)) else {
            die("could not allocate a write buffer")
        }
        buf.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer {
            buf.floatChannelData![0].update(from: $0.baseAddress!, count: x.count)
        }
        try file.write(from: buf)
    }

    static func f(_ v: Double, _ places: Int) -> String {
        String(format: "%.\(places)f", v)
    }

    static func die(_ msg: String) -> Never {
        FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
        exit(1)
    }
}
