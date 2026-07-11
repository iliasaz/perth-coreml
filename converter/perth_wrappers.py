"""Traceable, static-shape wrappers around Perth-Net Implicit's conv stacks.

Why wrappers exist
------------------
The ANE needs a *static* input shape, but audio length is dynamic. We therefore run the
conv stacks over fixed-width windows of W frames and tile them on the host.

Zero-padding a window is NOT equivalent to PyTorch's `padding=(k-1)//2` boundary
behaviour: `conv(zeros) = bias != 0`, so a zero-padded tail lights up after the first
layer and leaks back into the last HALO real frames at deeper layers. PyTorch instead
zero-pads the *activations* at every layer. We reproduce that exactly by multiplying the
activations by a validity mask after every layer -- which makes the windowed result
bit-identical to full-length PyTorch (verified: max|err| == 0.0).

Everything that is not a convolution (STFT/ISTFT, dB (de)normalisation, magmask,
interpolation, masked-mean, softmax, the residual add) stays on the host in fp32.
"""

from __future__ import annotations

import torch
from torch import nn

# 5 convs with k=7, padding=3 -> each extends the dependency cone by 3 frames per side.
HALO = 15


def _masked_forward(stack: nn.Sequential, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run `stack` with the activations zeroed outside `mask` after every layer.

    This is what makes a zero-padded static window exactly equal to the full-length
    PyTorch convolution -- see the module docstring.
    """
    h = x * mask
    for layer in stack:
        h = layer(h) * mask
    return h


class EncoderStack(nn.Module):
    """Perth encoder's residual conv stack, windowed.

    in : sub_mag (1, 128, W) -- the low 128 magnitude bins, dB-normalised
         mask    (1,   1, W) -- 1.0 on real frames, 0.0 on padding
    out: residual(1, 128, W) -- the watermark residual, BEFORE the magmask gate

    The magmask gate and the `magspec[:, :128] += res` add stay on the host: magmask needs
    all 1025 bins (this model only sees 128), and the add is free.
    """

    def __init__(self, encoder):
        super().__init__()
        self.layers = encoder.layers

    def forward(self, sub_mag, mask):
        return _masked_forward(self.layers, sub_mag, mask)


class DecoderStacks(nn.Module):
    """All three decoder branches in one graph -> one predict per tile.

    The branches are architecturally identical (Conv 128->256, 5x k=7, Conv 256->2) but
    carry different weights, so they cannot be batched; running them side by side in a
    single graph gives us one CoreML package and one ANE compile.

    in : {slow,norm,fast}_x (1, 128, W), {slow,norm,fast}_m (1, 1, W)
    out: {slow,norm,fast}_o (1,   2, W)   -- channel 0 = attention logit, 1 = watermark

    The host does the linear/nearest interpolation that produces each branch's input, the
    masked-mean pooling, the softmax over the three attention logits, and the combine.
    A branch that has run out of tiles is fed mask=0, which contributes 0 to both the
    numerator and the denominator of the masked mean -- so exhausted branches are free.
    """

    def __init__(self, decoder):
        super().__init__()
        self.slow_layers = decoder.slow_layers
        self.normal_layers = decoder.normal_layers
        self.fast_layers = decoder.fast_layers

    def forward(self, slow_x, slow_m, norm_x, norm_m, fast_x, fast_m):
        return (
            _masked_forward(self.slow_layers, slow_x, slow_m),
            _masked_forward(self.normal_layers, norm_x, norm_m),
            _masked_forward(self.fast_layers, fast_x, fast_m),
        )


def tile_plan(t: int, w: int, halo: int = HALO):
    """Windows needed to cover `t` frames with static width `w`.

    Yields (in_start, keep_lo, n_out, n_real): read input [in_start, in_start+w) (zero-padded
    to w if it runs past the end, with the mask 0 there), then keep n_out output frames
    starting at index keep_lo within the window.

    Interior windows have a fake cut at each edge, so their first and last `halo` outputs are
    wrong and get discarded -- hence a stride of w - 2*halo. The first window's left edge and
    the final window's right edge are *true* signal boundaries, where the model's own padding
    (plus the mask) is already correct, so nothing is discarded there.
    """
    stride = w - 2 * halo
    if stride <= 0:
        raise ValueError(f"window {w} too small for halo {halo}")
    out_start = 0
    while out_start < t:
        in_start = 0 if out_start == 0 else out_start - halo
        keep_lo = out_start - in_start                  # 0 on the first window, halo after
        n_real = min(w, t - in_start)                   # real frames this window holds
        n_out = min(stride if out_start else w - halo, t - out_start, n_real - keep_lo)
        yield in_start, keep_lo, n_out, n_real
        out_start += n_out
