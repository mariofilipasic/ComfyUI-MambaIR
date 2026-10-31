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


"""Reproducible routing, and the 1-channel colour policy.

MambaIRv2 routes every pixel with ``F.gumbel_softmax(logits, hard=True)``. That
**samples**, even in ``eval()``, so two identical runs differ visibly. It also
consumes the process-global RNG, which would make our output depend on whatever
ran before us in ComfyUI.

Both problems are solved the same way: an explicit :class:`torch.Generator`, held
in a context variable and used only while it is installed. Outside
:func:`seeded_router` the behaviour is bit-identical to upstream.
"""

from __future__ import annotations

import contextlib
import contextvars

import torch
import torch.nn.functional as F

__all__ = ["gumbel_hard", "seeded_router", "run_on_luma", "luma_of", "recombine_luma"]

_ROUTER_GEN: contextvars.ContextVar[torch.Generator | None] = contextvars.ContextVar(
    "mambair_router_generator", default=None
)


@contextlib.contextmanager
def seeded_router(seed: int | None, device=None):
    """Install a local generator for the duration of one forward pass.

    ``seed=None`` or ``seed < 0`` leaves the global RNG in charge (upstream
    behaviour). The global RNG is never read or written by :func:`gumbel_hard`.
    """
    if seed is None or int(seed) < 0:
        yield
        return
    gen = torch.Generator(device=device if device is not None else "cpu")
    gen.manual_seed(int(seed))
    token = _ROUTER_GEN.set(gen)
    try:
        yield
    finally:
        _ROUTER_GEN.reset(token)


def gumbel_hard(logits: torch.Tensor) -> torch.Tensor:
    """``F.gumbel_softmax(logits, hard=True, dim=-1)``, optionally generator-backed.

    Same distribution: sample Gumbel noise, take the arg-max, return the one-hot.
    """
    gen = _ROUTER_GEN.get()
    if gen is None:
        return F.gumbel_softmax(logits, hard=True, dim=-1)
    eps = torch.finfo(logits.dtype).tiny
    noise = torch.rand(
        logits.shape, generator=gen, device=logits.device, dtype=logits.dtype
    ).clamp_(eps, 1.0)
    noise = -torch.log(-torch.log(noise))
    index = (logits + noise).argmax(dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, index, 1.0)


# --------------------------------------------------------------------------- #
# 1-channel checkpoints (MambaIR JPEG-CAR) applied to RGB images
# --------------------------------------------------------------------------- #
def luma_of(x: torch.Tensor) -> torch.Tensor:
    """Rec.601 luma of an ``(N, 3, H, W)`` tensor in ``[0, 1]``."""
    return (
        0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    )


def recombine_luma(restored: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Keep the input's chroma and take the model's luma.

    ``R' = Y' + (R - Y)`` etc. is exactly "chroma unchanged": ``R - Y`` *is* the
    red-difference chroma signal.
    """
    y0 = luma_of(original)
    offsets = original - y0
    return restored + offsets


def run_on_luma(fn, x: torch.Tensor) -> torch.Tensor:
    """Apply a 1-channel model to the luma of an RGB tensor."""
    restored = fn(luma_of(x))
    if restored.shape[1] != 1:
        restored = restored[:, :1]
    return recombine_luma(restored, x)
