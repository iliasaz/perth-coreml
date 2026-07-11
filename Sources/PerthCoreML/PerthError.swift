import Foundation

public enum PerthError: Error, CustomStringConvertible {
    /// The STFT reflect-pads by `nFFT/2`, which needs at least that many samples to mirror.
    /// Python throws here too; we surface it rather than crashing.
    case signalTooShort(samples: Int, minimum: Int)
    case modelNotFound(name: String, directory: URL)
    case badModelOutput(String)
    case unsupportedSampleRate(Int)

    public var description: String {
        switch self {
        case let .signalTooShort(n, minimum):
            return "signal too short: \(n) samples at 32 kHz, need at least \(minimum) "
                 + "(the STFT reflect-pads by \(PerthConfig.nFFT / 2))"
        case let .modelNotFound(name, dir):
            return "CoreML model '\(name)' not found in \(dir.path)"
        case let .badModelOutput(what):
            return "unexpected CoreML output: \(what)"
        case let .unsupportedSampleRate(sr):
            return "unsupported sample rate \(sr); built-in resampling covers 24 kHz and 32 kHz"
        }
    }
}
