import Foundation

/// Perth-Net Implicit's hyperparameters, from `pretrained/implicit/hparams.yaml`.
///
/// These are not tunable: they are baked into the trained weights. `subband` in particular is
/// derived, not stored -- `round(1025 * 2000 / 16000) == 128`.
public enum PerthConfig {
    public static let sampleRate = 32_000
    public static let nFFT = 2_048
    public static let hopSize = 320
    public static let windowSize = 2_048
    public static let nFreq = nFFT / 2 + 1        // 1025
    public static let subband = 128               // round(nFreq * maxWmarkFreq / (sampleRate/2))

    /// `20 * log10(1e-9)`.
    static let minLevelDB: Float = -180
    /// `-minLevelDB + headroomDB(15)`. Both constants are exact in fp32.
    static let denormScale: Float = 195
    /// magmask keeps frames whose total energy exceeds this fraction of the loudest frame.
    static let magmaskP: Float = 0.05

    /// Static frame width of the CoreML conv stacks. The ANE needs a fixed shape; the host
    /// tiles to cover arbitrary lengths. Must match the `--window` the converter was run with.
    public static let modelWindow = 1_024
    /// Frames of context each tile needs on a cut edge: 5 convs of k=7, padding 3 => 5*3.
    static let halo = 15

    /// The shortest signal the STFT accepts. `center=true` reflect-pads by `nFFT/2`, and reflect
    /// padding needs at least that many samples to mirror; PyTorch throws below this.
    public static let minSamples32k = nFFT / 2 + 1     // 1025
}
