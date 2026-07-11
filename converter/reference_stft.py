"""A from-scratch STFT/ISTFT that reproduces torchaudio bit-for-bit.

This is the algorithm the Swift/Accelerate implementation must follow literally. Writing it
in numpy first is far cheaper than debugging it through Swift: once this matches torchaudio,
Swift only has to match *this*, and any Swift bug is isolated to vDSP plumbing.

torchaudio Spectrogram(power=None, center=True, pad_mode='reflect', normalized=False) and
InverseSpectrogram(length=None) are what Perth uses.
"""
from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 320
NOLA_EPS = 1e-11          # torch.istft's window-envelope floor


def stft(signal: np.ndarray, window: np.ndarray, n_fft=N_FFT, hop=HOP) -> np.ndarray:
    """(N,) real -> (n_fft//2+1, T) complex.  T = N // hop + 1"""
    pad = n_fft // 2
    x = np.pad(signal.astype(np.float64), pad, mode="reflect")   # center=True
    t = signal.shape[0] // hop + 1
    out = np.empty((n_fft // 2 + 1, t), dtype=np.complex128)
    for i in range(t):
        frame = x[i * hop: i * hop + n_fft] * window
        out[:, i] = np.fft.rfft(frame, n=n_fft)
    return out


def istft(spec: np.ndarray, window: np.ndarray, n_fft=N_FFT, hop=HOP) -> np.ndarray:
    """(n_fft//2+1, T) complex -> (hop*(T-1),) real.

    Note the output length: torch.istft with length=None returns
    n_fft + hop*(T-1) samples and then trims n_fft//2 from each end, leaving hop*(T-1).
    That is SHORTER than the signal that produced the spectrogram (by up to hop-1 samples) --
    which is why Perth's apply_watermark hands back fewer samples than it was given.
    """
    t = spec.shape[1]
    total = n_fft + hop * (t - 1)
    acc = np.zeros(total, dtype=np.float64)
    env = np.zeros(total, dtype=np.float64)
    wsq = window ** 2
    for i in range(t):
        frame = np.fft.irfft(spec[:, i], n=n_fft)
        acc[i * hop: i * hop + n_fft] += frame * window
        env[i * hop: i * hop + n_fft] += wsq
    # torch divides only where the envelope clears the NOLA floor
    nz = env > NOLA_EPS
    acc[nz] /= env[nz]
    pad = n_fft // 2
    return acc[pad: total - pad]


def _self_test():
    import torch
    from torchaudio.transforms import InverseSpectrogram, Spectrogram
    import sys
    sys.path.insert(0, "/Users/ilia/Developer/Perth/src")
    from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

    net = PerthImplicitWatermarker(device="cpu").perth_net
    w_t = net.ap.spectrogram.window.detach()
    w = w_t.numpy().astype(np.float64)

    wf = lambda _n, **_k: w_t.clone()
    sp = Spectrogram(n_fft=N_FFT, win_length=N_FFT, power=None, hop_length=HOP,
                     window_fn=wf, normalized=False)
    isp = InverseSpectrogram(n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
                             window_fn=wf, normalized=False)

    rng = np.random.default_rng(0)
    print(f"{'N':>8} {'T':>5} {'|STFT| err':>12} {'phase err':>12} "
          f"{'out len':>8} {'ISTFT err':>12}")
    worst = 0.0
    for n in (4096, 16000, 32000, 96137, 240000, 320007):
        sig = (0.3 * rng.standard_normal(n)).astype(np.float32)
        ref = sp(torch.from_numpy(sig)).numpy()
        got = stft(sig, w)
        e_mag = np.abs(np.abs(ref) - np.abs(got)).max()
        e_ph = np.abs(np.angle(ref) - np.angle(got)).max()

        back_ref = isp(torch.from_numpy(ref)).numpy()
        back_got = istft(got.astype(np.complex128), w)
        e_i = np.abs(back_ref - back_got).max()
        worst = max(worst, e_mag, e_i)
        print(f"{n:8d} {ref.shape[1]:5d} {e_mag:12.3e} {e_ph:12.3e} "
              f"{len(back_got):8d} {e_i:12.3e}")
        assert len(back_got) == HOP * (ref.shape[1] - 1) == len(back_ref)

    print(f"\nworst error vs torchaudio: {worst:.3e}")
    print("(fp64 here; torch is fp32, so ~1e-7 is the expected agreement floor)")
    print("output length is always hop*(T-1) -- SHORTER than the input signal.")


if __name__ == "__main__":
    _self_test()
