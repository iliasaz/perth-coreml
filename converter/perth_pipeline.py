"""The host-side half of the port, in Python.

This is the reference the Swift implementation must reproduce. It performs every operation
that stays off the ANE -- STFT/ISTFT, dB (de)normalisation, magmask, interpolation,
masked-mean, softmax, the residual add, and the windowed tiling -- and delegates only the
conv stacks to a pluggable `runner` (PyTorch or CoreML).

Running this with the PyTorch runner must reproduce stock Perth bit-for-bit; that isolates
"did we decompose the model correctly" from "did CoreML convert it faithfully".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import InverseSpectrogram, Spectrogram

from perth_wrappers import HALO, tile_plan

SUBBAND = 128
HEADROOM_DB = 15.0
STFT_MAGNITUDE_MIN = 1e-9
MIN_LEVEL_DB = 20.0 * np.log10(STFT_MAGNITUDE_MIN)   # -180.0
DENORM_SCALE = -MIN_LEVEL_DB + HEADROOM_DB           # 195.0


class AudioFront:
    """torchaudio STFT/ISTFT + Perth's dB normalisation. Matches AudioProcessor exactly.

    `window` MUST be the window from the checkpoint, not a fresh torch.hann_window(2048).
    Perth's AudioProcessor is an nn.Module, so `ap.spectrogram.window` is a registered
    buffer that load_state_dict OVERWRITES with the value stored in perth_net_250000.pth.tar.
    That stored window differs from today's torch.hann_window by 1 ULP (5.96e-08) -- enough
    to put a ~2e-7 floor under every waveform comparison until you use the checkpoint's.
    """

    def __init__(self, window: torch.Tensor, n_fft=2048, hop=320, win=2048):
        assert window.shape == (win,), f"expected a {win}-tap window, got {tuple(window.shape)}"
        wf = lambda _n, **_kw: window.clone()
        self.spec = Spectrogram(n_fft=n_fft, win_length=win, power=None, hop_length=hop,
                                window_fn=wf, normalized=False)
        self.ispec = InverseSpectrogram(n_fft=n_fft, win_length=win, hop_length=hop,
                                        window_fn=wf, normalized=False)

    def to_magphase(self, signal: torch.Tensor):
        spec = self.spec(signal.float())
        phase = torch.angle(spec)
        mag = spec.abs()
        mag = 20 * torch.log10(mag.clip(STFT_MAGNITUDE_MIN))
        mag = (mag - MIN_LEVEL_DB) / DENORM_SCALE
        return mag, phase

    def to_signal(self, mag: torch.Tensor, phase: torch.Tensor):
        mag = mag * DENORM_SCALE + MIN_LEVEL_DB
        mag = 10.0 ** ((mag / 20).clip(max=10))
        return self.ispec(mag * torch.exp(1.0j * phase))


def magmask(magspec: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    """(B, 1025, T) -> (B, 1, T). Frames whose total energy exceeds 5% of the loudest frame.

    Needs all 1025 bins, so it cannot live inside the CoreML model (which only sees 128).
    """
    s = magspec.sum(dim=1)
    thresh = s.max(dim=1).values * p
    return (s > thresh[:, None]).float()[:, None]


class PerthPipeline:
    """apply / detect, with the conv stacks delegated to `runner`.

    runner.encode(x, m)                       -> (1, 128, W)
    runner.decode(sx, sm, nx, nm, fx, fm)     -> three (1, 2, W)
    Both take and return numpy float32 arrays of static width `runner.window`.
    """

    def __init__(self, runner, window: torch.Tensor):
        self.runner = runner
        self.front = AudioFront(window)
        self.W = runner.window

    # -- tiling helpers ----------------------------------------------------------------
    def _windows(self, t: int):
        return list(tile_plan(t, self.W, HALO))

    def _pad_window(self, x: np.ndarray, in_start: int, n_real: int):
        """Slice [in_start, in_start+W) out of x, zero-padded, plus its validity mask."""
        c = x.shape[1]
        buf = np.zeros((1, c, self.W), dtype=np.float32)
        buf[:, :, :n_real] = x[:, :, in_start:in_start + n_real]
        m = np.zeros((1, 1, self.W), dtype=np.float32)
        m[:, :, :n_real] = 1.0
        return buf, m

    # -- watermark embedding -----------------------------------------------------------
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """32 kHz float32 in -> watermarked 32 kHz float32 out.

        Note the output is (T-1)*hop samples, which is generally SHORTER than the input --
        that is stock Perth's behaviour (istft drops the trailing partial frame) and we
        reproduce it rather than "fixing" it.
        """
        mag, phase = self.front.to_magphase(torch.from_numpy(signal))
        mag = mag[None]                                   # (1, 1025, T)
        gate = magmask(mag)                               # (1, 1, T)
        sub = mag[:, :SUBBAND].numpy()                    # (1, 128, T)
        t = sub.shape[2]

        res = np.zeros_like(sub)
        for in_start, keep_lo, n_out, n_real in self._windows(t):
            x, m = self._pad_window(sub, in_start, n_real)
            y = self.runner.encode(x, m)
            out_start = in_start + keep_lo
            res[:, :, out_start:out_start + n_out] = y[:, :, keep_lo:keep_lo + n_out]

        res = torch.from_numpy(res) * gate                # the semantic magmask gate
        wm = mag.clone()
        wm[:, :SUBBAND] += res
        return self.front.to_signal(wm[0], phase).numpy()

    # -- watermark detection -----------------------------------------------------------
    def detect(self, signal: np.ndarray, do_round: bool = True) -> float:
        mag, _ = self.front.to_magphase(torch.from_numpy(signal))
        mag = mag[None]
        gate = magmask(mag)                               # (1, 1, T)
        sub = mag[:, :SUBBAND]                            # (1, 128, T)
        t = sub.shape[2]

        lens = {"slow": int(t * 1.25), "norm": t, "fast": int(t * 0.75)}
        xs, gates = {}, {}
        for k, ti in lens.items():
            xs[k] = (sub if k == "norm"
                     else F.interpolate(sub, size=ti, mode="linear", align_corners=True)).numpy()
            # nearest-interpolate the semantic mask to the branch's length
            gates[k] = F.interpolate(gate, size=ti, mode="nearest").numpy()

        plans = {k: self._windows(lens[k]) for k in lens}
        n_tiles = max(len(p) for p in plans.values())

        # masked-mean accumulators: sum(x*m) and sum(m) accumulate linearly across tiles
        num = {k: np.zeros(2, dtype=np.float64) for k in lens}   # [attn, wmark]
        den = {k: 0.0 for k in lens}

        for i in range(n_tiles):
            feed = {}
            for k in ("slow", "norm", "fast"):
                if i < len(plans[k]):
                    in_start, keep_lo, n_out, n_real = plans[k][i]
                    feed[k] = self._pad_window(xs[k], in_start, n_real)
                else:
                    # exhausted branch: mask=0 contributes nothing to numerator or denominator
                    feed[k] = (np.zeros((1, 128, self.W), np.float32),
                               np.zeros((1, 1, self.W), np.float32))
            outs = self.runner.decode(feed["slow"][0], feed["slow"][1],
                                      feed["norm"][0], feed["norm"][1],
                                      feed["fast"][0], feed["fast"][1])
            for k, o in zip(("slow", "norm", "fast"), outs):
                if i >= len(plans[k]):
                    continue
                in_start, keep_lo, n_out, n_real = plans[k][i]
                out_start = in_start + keep_lo
                seg = o[0, :, keep_lo:keep_lo + n_out]                        # (2, n_out)
                g = gates[k][0, 0, out_start:out_start + n_out]               # (n_out,)
                num[k] += (seg.astype(np.float64) * g).sum(axis=1)
                den[k] += float(g.sum())

        # Silent audio makes magmask all-zero, so den == 0 and the masked mean is 0/0 -> NaN.
        # Stock Perth returns NaN there; we reproduce it rather than "fixing" it.
        with np.errstate(invalid="ignore", divide="ignore"):
            attn = np.array([num[k][0] / den[k] for k in ("slow", "norm", "fast")])
            wmks = np.array([num[k][1] / den[k] for k in ("slow", "norm", "fast")])
        a = np.exp(attn - attn.max())
        a /= a.sum()
        conf = float((wmks * a).sum())
        conf = float(np.clip(conf, 0.0, 1.0))          # NaN-propagating, like torch.clip
        if do_round:
            conf = float(np.round(conf))               # round-half-to-EVEN, like torch.round
        return conf


class TorchRunner:
    """Runs the stacks with the traced-but-not-converted PyTorch wrappers."""

    def __init__(self, net, window: int):
        from perth_wrappers import DecoderStacks, EncoderStack
        self.window = window
        self.enc = EncoderStack(net.encoder).eval()
        self.dec = DecoderStacks(net.decoder).eval()

    @torch.no_grad()
    def encode(self, x, m):
        return self.enc(torch.from_numpy(x), torch.from_numpy(m)).numpy()

    @torch.no_grad()
    def decode(self, sx, sm, nx, nm, fx, fm):
        o = self.dec(*(torch.from_numpy(a) for a in (sx, sm, nx, nm, fx, fm)))
        return tuple(t.numpy() for t in o)
