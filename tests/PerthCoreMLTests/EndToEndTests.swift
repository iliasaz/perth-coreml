import CoreML
import Foundation
import Testing

@testable import PerthCoreML

/// Where the `.mlpackage`s live. The suite skips itself when this is unset, so `swift test` stays
/// green on a machine that has never run the converter.
private let modelDirectory: URL? = ProcessInfo.processInfo.environment["PERTH_MODEL_DIR"]
    .map { URL(fileURLWithPath: $0) }

@Suite("End to end", .enabled(if: modelDirectory != nil,
                              "set PERTH_MODEL_DIR to the directory holding the .mlpackages"))
struct EndToEndTests {
    let watermarker: PerthWatermarker

    init() throws {
        watermarker = try PerthWatermarker(modelDirectory: modelDirectory!, computeUnits: .all)
    }

    @Test("a watermarked 32 kHz signal is detected and a clean one is not")
    func roundTrip32k() throws {
        let sr = PerthConfig.sampleRate
        let clean = harmonicSignal(sr * 2, sampleRate: sr)

        let marked = try watermarker.applyWatermark(clean, sampleRate: sr)
        #expect(marked.count == clean.count, ".preserveLength must hand back every sample")

        let raw = try watermarker.getWatermark(marked, sampleRate: sr, round: false)
        let cleanRaw = try watermarker.getWatermark(clean, sampleRate: sr, round: false)
        #expect(try watermarker.getWatermark(marked, sampleRate: sr) == 1,
                "watermarked: raw score \(raw)")
        #expect(try watermarker.getWatermark(clean, sampleRate: sr) == 0,
                "clean: raw score \(cleanRaw)")

        // A "the residual did not explode" guard, not an imperceptibility claim: stock Python Perth
        // measures 13.6 dB on this exact signal (and 14.9 dB on real speech), so the watermark is
        // simply not a quiet one. The bound is there to catch a residual applied to the wrong bins
        // or with the gate inverted, both of which cost tens of dB.
        let db = snr(clean, marked)
        #expect(db > 10, "watermark SNR \(db) dB -- python measures 13.6 dB here")
    }

    /// 24 kHz is the chatterbox path, and it is the one that runs the resamplers -- soxr_hq on the
    /// way in, `resample_poly` on the way back out for detection.
    @Test("the 24 kHz path resamples, watermarks and still detects")
    func roundTrip24k() throws {
        let sr = 24_000
        let clean = harmonicSignal(sr * 2, sampleRate: sr, seed: 5)

        let marked = try watermarker.applyWatermark(clean, sampleRate: sr)
        #expect(marked.count == clean.count)

        let raw = try watermarker.getWatermark(marked, sampleRate: sr, round: false)
        #expect(try watermarker.getWatermark(marked, sampleRate: sr) == 1,
                "watermarked: raw score \(raw)")
        #expect(try watermarker.getWatermark(clean, sampleRate: sr) == 0)
    }

    /// A signal too short to reflect-pad is handed back untouched rather than throwing, so a caller
    /// stitching short chunks never crashes mid-stream.
    @Test("a signal shorter than the reflect pad passes through unchanged")
    func tooShortPassesThrough() throws {
        let x = harmonicSignal(500, sampleRate: PerthConfig.sampleRate)
        let out = try watermarker.applyWatermark(x, sampleRate: PerthConfig.sampleRate)
        #expect(out == x)
    }

    @Test("detection on a too-short signal throws")
    func detectTooShort() {
        let x = harmonicSignal(500, sampleRate: PerthConfig.sampleRate)
        #expect(throws: PerthError.self) {
            _ = try watermarker.getWatermark(x, sampleRate: PerthConfig.sampleRate)
        }
    }

    @Test("an unsupported sample rate throws rather than silently mis-resampling")
    func unsupportedRate() {
        let x = harmonicSignal(16_000, sampleRate: 16_000)
        #expect(throws: PerthError.self) {
            _ = try watermarker.applyWatermark(x, sampleRate: 16_000)
        }
        #expect(throws: PerthError.self) {
            _ = try watermarker.getWatermark(x, sampleRate: 16_000)
        }
    }

    /// Digital silence makes magmask all-zero, so the masked mean is 0/0. Stock Perth returns NaN;
    /// anything else would be inventing an answer.
    @Test("silence detects as NaN, not as zero")
    func silenceIsNaN() throws {
        let x = [Float](repeating: 0, count: 32_000)
        let raw = try watermarker.getWatermark(x, sampleRate: PerthConfig.sampleRate, round: false)
        #expect(raw.isNaN, "got \(raw)")
    }

    /// A signal longer than the model's static window forces the host to tile. The seam is where a
    /// halo or coverage bug would show up as a periodic artefact.
    @Test("a signal long enough to need several tiles still detects")
    func multiTile() throws {
        let sr = PerthConfig.sampleRate
        // 3 encoder tiles: T = n/320 + 1 must exceed 2*(1024 - 30).
        let n = 2500 * PerthConfig.hopSize
        #expect(tilePlan(frames: STFT.frameCount(n)).count >= 3)

        let clean = harmonicSignal(n, sampleRate: sr, seed: 8)
        let marked = try watermarker.applyWatermark(clean, sampleRate: sr)
        let raw = try watermarker.getWatermark(marked, sampleRate: sr, round: false)
        #expect(try watermarker.getWatermark(marked, sampleRate: sr) == 1,
                "watermarked: raw score \(raw)")
        #expect(try watermarker.getWatermark(clean, sampleRate: sr) == 0)
    }

    @Test("pythonParity truncates the way Perth's ISTFT does")
    func pythonParityLength() throws {
        let parity = try PerthWatermarker(modelDirectory: modelDirectory!,
                                          computeUnits: .all,
                                          lengthPolicy: .pythonParity)
        let n = 32_000 + 137                        // not a whole number of hops
        let clean = harmonicSignal(n, sampleRate: PerthConfig.sampleRate)
        let marked = try parity.applyWatermark(clean, sampleRate: PerthConfig.sampleRate)

        let t = STFT.frameCount(n)
        #expect(marked.count == STFT.signalLength(frames: t))
        #expect(marked.count < n)
    }
}
