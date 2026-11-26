"""Reference for the megakernel SO(2) conv operator (C8).

Two implementations of the same formula, deliberately:

``so2_conv_dmk_kernel_order``
    A literal transcription of the operator's per-edge loop, in the same index
    order and with the same fma chain, in float64.  It is the oracle the DCU
    test compares against, so if the kernel is changed the oracle changes with
    it -- and if the kernel's *math* is wrong, this one is wrong in exactly the
    same way, which is why the second implementation exists.

``so2_conv_dmk_reference``
    The same math written as batched matmuls over the whole edge array.  It
    shares no code with the loop, so a bug in the transcription shows up as a
    disagreement between the two rather than as a silent pass.

The stages, for one edge ``e`` and channel ``c``::

    xl[r]     = sum_d  R[e][r][d] * x[src[e]][d][c]         rotate to local frame
    for layer in 0..L-1:
      z[r]    = sum_r' K[e][layer][0][r][r'] * xl[r']
      gate[r] = sum_r' K[e][layer][1][r][r'] * xl[r']
      xl[r]  += silu(z[r]) * sigmoid(gate[r])
    msg[e][d] = alpha[e] * sum_r R_t[e][d][r] * xl[r]       rotate back + weight

The destination reduction is not part of the operator (see
``docs/so2-conv-dmk.md``): ``msg`` is per edge, and summing it onto atoms is the
separate segment-sum instruction.
"""

from __future__ import annotations

import math

import torch


def so2_conv_dmk_kernel_order(
    x: torch.Tensor,
    src: torch.Tensor,
    rot: torch.Tensor,
    rot_t: torch.Tensor,
    kc: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Transcribe the operator's loop, one edge and one channel at a time.

    Parameters
    ----------
    x
        ``(n_node, dim, channels)`` node features.
    src
        ``(n_edge,)`` source node of each edge.
    rot
        ``(n_edge, red, dim)`` rotation into the local frame.
    rot_t
        ``(n_edge, dim, red)`` its transpose (the inverse rotation).
    kc
        ``(n_edge, layers, 2, red, red)`` content kernel then gate kernel.
    alpha
        ``(n_edge,)`` envelope / attention weight.

    Returns
    -------
    ``(n_edge, dim, channels)`` per-edge message, in float64.
    """
    edges = int(src.shape[0])
    dim = int(rot.shape[2])
    red = int(rot.shape[1])
    channels = int(x.shape[2])
    layers = int(kc.shape[1])

    x64 = x.to(torch.float64)
    r64 = rot.to(torch.float64)
    rt64 = rot_t.to(torch.float64)
    k64 = kc.to(torch.float64)
    a64 = alpha.to(torch.float64)
    out = torch.zeros(edges, dim, channels, dtype=torch.float64, device=x.device)

    for e in range(edges):
        node = int(src[e])
        for c in range(channels):
            xl = [0.0] * red
            for r in range(red):
                acc = 0.0
                for d in range(dim):
                    acc += r64[e, r, d] * x64[node, d, c]
                xl[r] = acc

            for layer in range(layers):
                z = [0.0] * red
                gate = [0.0] * red
                for r in range(red):
                    a = 0.0
                    b = 0.0
                    for rp in range(red):
                        v = xl[rp]
                        a += k64[e, layer, 0, r, rp] * v
                        b += k64[e, layer, 1, r, rp] * v
                    z[r] = a
                    gate[r] = b
                for r in range(red):
                    # silu(z) * sigmoid(gate), the gated layer's residual term.
                    sigmoid_z = 1.0 / (1.0 + math.exp(-z[r]))
                    sigmoid_g = 1.0 / (1.0 + math.exp(-gate[r]))
                    xl[r] = xl[r] + (z[r] * sigmoid_z) * sigmoid_g

            for d in range(dim):
                acc = 0.0
                for r in range(red):
                    acc += rt64[e, d, r] * xl[r]
                out[e, d, c] = a64[e] * acc
    return out


def so2_conv_dmk_reference(
    x: torch.Tensor,
    src: torch.Tensor,
    rot: torch.Tensor,
    rot_t: torch.Tensor,
    kc: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """The same math as batched matmuls, sharing no code with the loop above."""
    x64 = x.to(torch.float64)
    rows = x64[src]  # (E, dim, C)
    # Rotate: (E, red, dim) @ (E, dim, C)
    xl = torch.einsum("erd,edc->erc", rot.to(torch.float64), rows)
    for layer in range(kc.shape[1]):
        # Note the distinct inner subscripts: `"err,erc->erc"` would read the
        # kernel's *diagonal* (torch treats a repeated input subscript that
        # way), which is the kind of slip the loop below exists to catch.
        content = torch.einsum(
            "ers,esc->erc", kc[:, layer, 0].to(torch.float64), xl
        )
        gate = torch.einsum("ers,esc->erc", kc[:, layer, 1].to(torch.float64), xl)
        xl = xl + torch.nn.functional.silu(content) * torch.sigmoid(gate)
    out = torch.einsum("edr,erc->edc", rot_t.to(torch.float64), xl)
    return out * alpha.to(torch.float64).view(-1, 1, 1)
