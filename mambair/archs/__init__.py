# Derived from basicsr.archs.arch_util and timm.models.layers (both Apache-2.0).
# Licence: Apache-2.0 -- see PROVENANCE.md and LICENSE-APACHE-2.0. Not part of this pack's
# AGPL-3.0-or-later work, so this file carries no AGPL header (tests/test_license.py).

"""Helpers the vendored MambaIR architectures need.

Upstream imports these from ``basicsr.archs.arch_util`` and ``timm.models.layers``.
Both are plain, tiny utilities, so they are vendored here instead of adding two
dependencies (``basicsr`` is not installable without dragging in a whole training
framework, and ``timm.models.layers`` is deprecated in timm >= 1.0).
"""

from __future__ import annotations

import collections.abc
from itertools import repeat as _repeat

import torch
from torch.nn.init import trunc_normal_ as _trunc_normal

__all__ = ["to_2tuple", "trunc_normal_", "DropPath"]


def to_2tuple(x):
    """Upstream: ``basicsr.archs.arch_util.to_2tuple`` / ``timm.models.layers.to_2tuple``."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(_repeat(x, 2))


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """``timm.models.layers.trunc_normal_`` is ``torch.nn.init.trunc_normal_``."""
    return _trunc_normal(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth (timm ``DropPath``). Inference is a no-op."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(torch.nn.Module):
    """Used by the MambaIR (v1) architecture file."""

    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
