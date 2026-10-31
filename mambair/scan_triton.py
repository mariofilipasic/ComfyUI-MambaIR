# MambaIR for ComfyUI
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mario Filipasic
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version. It is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of it with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""Fused Triton scan (tier 1).

Imported lazily by :mod:`mambair.scan`; a missing or broken ``triton`` install makes
this module unimportable and the dispatch stays on the PyTorch tier.

How the recurrence is parallelised
----------------------------------
The recurrence ``x_t = a_t * x_{t-1} + b_t`` is sequential along the sequence but independent per
``(batch, channel)``, and its step composes associatively::

    (a1, b1) then (a2, b2)  ==  (a1 * a2, a2 * b1 + b2)

so a *prefix scan* over a block of tokens is one ``tl.associative_scan`` away, with the state vector
held in registers. That is what this module does: one program per row walks its sequence in blocks
of ``BT`` tokens, and only the state crossing a block boundary is carried -- also in a register.
Nothing of shape ``(batch, channels, length, state)`` is ever built, and there is exactly one sweep
over the sequence, with no per-chunk carry array.

The dependent chain is what costs: an implementation that advances one token per loop iteration
stalls on memory latency regardless of how many programs it launches. Scanning ``BT`` tokens per
step shortens that chain by ``BT`` and makes every load ``BT`` tokens wide. Throughput numbers live
in ``README.md``; regenerate them with ``python tests/bench.py`` instead of editing them here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Tokens per block: how many tokens one associative scan covers, and how wide each load is.
BT = 32

#: Warps per program. Two covers the ``(BT, BLOCK_N)`` tile and its temporaries without spilling.
NUM_WARPS = 2


def _block_tokens(block_n: int) -> int:
    """Return the block length for a given state width.

    The tile is ``(BT, block_n)`` fp32 values plus a handful of live temporaries, so a wide
    ``d_state`` has to buy its registers back from the token block rather than spill.
    """
    return BT if block_n <= 64 else BT // 2


# The scan's associative operator. Triton parses the body of a kernel, so it carries comments
# instead of a docstring (see docs/docstring-waivers.md).
@triton.jit
def _combine(a1, b1, a2, b2):
    # (a1, b1) then (a2, b2): x -> a2 * (a1 * x + b1) + b2
    return a1 * a2, a2 * b1 + b2


# One program per (batch, channel) row: prefix-scan BT tokens at a time, carry the state.
@triton.jit
def _row_scan(u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, bias_ptr, D_ptr, z_ptr, y_ptr, last_ptr,
              L, N, D, G, stride_ub, stride_ud, stride_bb, stride_bg, stride_zb, stride_zd,
              HAS_BIAS: tl.constexpr, SOFTPLUS: tl.constexpr, HAS_D: tl.constexpr,
              HAS_Z: tl.constexpr, BT: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    bi = row // D
    di = row % D
    gi = di // (D // G)

    cols = tl.arange(0, BLOCK_N)
    nmask = cols < N
    A = tl.load(A_ptr + di * N + cols, mask=nmask, other=0.0)
    bias = tl.load(bias_ptr + di) if HAS_BIAS else 0.0
    Dv = tl.load(D_ptr + di) if HAS_D else 0.0

    tt = tl.arange(0, BT)
    x = tl.zeros([BLOCK_N], dtype=tl.float32)
    for t0 in range(0, L, BT):
        ts = t0 + tt
        tmask = ts < L
        base = bi * stride_ub + di * stride_ud + ts
        dt = tl.load(delta_ptr + base, mask=tmask, other=0.0) + bias
        if SOFTPLUS:
            dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
        uu = tl.load(u_ptr + base, mask=tmask, other=0.0)
        bofs = bi * stride_bb + gi * stride_bg + ts[:, None] * N + cols[None, :]
        m2 = tmask[:, None] & nmask[None, :]
        bb = tl.load(B_ptr + bofs, mask=m2, other=0.0)
        a = tl.exp(dt[:, None] * A[None, :])
        q = dt[:, None] * bb * uu[:, None]
        # inclusive prefix scan of the block; (ap, qp) is the state at each token from a zero start
        ap, qp = tl.associative_scan((a, q), 0, _combine)
        xs = qp + ap * x[None, :]
        cc = tl.load(C_ptr + bofs, mask=m2, other=0.0)
        yt = tl.sum(xs * cc, axis=1)
        if HAS_D:
            yt = yt + uu * Dv
        if HAS_Z:
            zt = tl.load(z_ptr + bi * stride_zb + di * stride_zd + ts, mask=tmask, other=0.0)
            yt = yt * (zt / (1.0 + tl.exp(-zt)))
        tl.store(y_ptr + row * L + ts, yt, mask=tmask)
        # Carry the state after the block's last *valid* token: a ragged tail must not carry a
        # masked lane (whose `a` is softplus(bias) and whose `q` is zero).
        steps = L - t0
        last_i = tl.where(steps > BT, BT - 1, steps - 1)
        x = tl.sum(tl.where((tt == last_i)[:, None], xs, 0.0), axis=0)

    tl.store(last_ptr + row * N + cols, x, mask=nmask)


def selective_scan_triton(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                          delta_softplus=False, return_last_state=False):
    """Run the fused scan. Drop-in for ``mamba_ssm``'s function of the same name.

    One program per ``(batch, channel)`` row prefix-scans ``BT`` tokens at a time with an
    associative scan, so the recurrence's parallelism is exposed instead of being walked one token
    per loop iteration.

    Args:
        u: Input, shape ``(batch, channels, length)``.
        delta: Step sizes, shape ``(batch, channels, length)``.
        A: State transition, shape ``(channels, d_state)``.
        B: Input projection, ``(batch, groups, d_state, length)`` or ``(batch, d_state, length)``.
        C: Output projection, same layout as ``B``.
        D: Per-channel skip weight, shape ``(channels,)``, or ``None``.
        z: Gate for the SiLU-multiplied variant, ``(batch, channels, length)``, or ``None``.
        delta_bias: Per-channel bias added before ``softplus``, shape ``(channels,)``.
        delta_softplus: Apply ``softplus`` to ``delta``.
        return_last_state: Also return the final state.

    Returns:
        Output of shape ``(batch, channels, length)``, or ``(output, last_state)``.

    Raises:
        ValueError: If ``groups`` does not divide ``channels``.
        NotImplementedError: If ``d_state`` exceeds 128, which the tile cannot hold in registers.
    """
    b, d, l = u.shape
    n = A.shape[1]
    B4 = B if B.dim() == 4 else B.unsqueeze(1)  # (b, g, n, l)
    C4 = C if C.dim() == 4 else C.unsqueeze(1)
    groups = B4.shape[1]
    if d % groups:
        raise ValueError(f"groups ({groups}) must divide channels ({d})")
    if n > 128:
        raise NotImplementedError(f"d_state={n} is too large for the fused scan")

    u = u.contiguous().float()
    delta = delta.contiguous().float()
    A = A.contiguous().float()
    # (b, g, l, n) so one token's states are a single contiguous load
    B4 = B4.transpose(-1, -2).contiguous().float()
    C4 = C4.transpose(-1, -2).contiguous().float()
    if delta_bias is None:
        bias = torch.zeros(1, device=u.device, dtype=torch.float32)
        has_bias = False
    else:
        bias = delta_bias.float().contiguous()
        has_bias = True
    Dv = torch.zeros(d, device=u.device, dtype=torch.float32) if D is None else D.float().contiguous()
    if z is None:
        zv = torch.zeros(1, 1, 1, device=u.device, dtype=torch.float32)
        stride_zb = stride_zd = 0
        has_z = False
    else:
        zv = z.contiguous().float()
        stride_zb, stride_zd = zv.stride(0), zv.stride(1)
        has_z = True

    rows = b * d
    y = torch.empty(rows, l, device=u.device, dtype=torch.float32)
    last = torch.empty(rows, n, device=u.device, dtype=torch.float32)
    blk = max(16, triton.next_power_of_2(n))
    _row_scan[(rows,)](u, delta, A, B4, C4, bias, Dv, zv, y, last,
                       l, n, d, groups, u.stride(0), u.stride(1),
                       B4.stride(0), B4.stride(1), stride_zb, stride_zd,
                       HAS_BIAS=has_bias, SOFTPLUS=bool(delta_softplus), HAS_D=D is not None,
                       HAS_Z=has_z, BT=_block_tokens(blk), BLOCK_N=blk, num_warps=NUM_WARPS)

    y = y.view(b, d, l)
    if return_last_state:
        return y, last.view(b, d, n)
    return y
