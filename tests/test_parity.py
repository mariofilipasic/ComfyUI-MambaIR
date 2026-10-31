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


"""P-PARITY-1: the vendored architecture is the upstream one.

The test process injects stand-ins for ``basicsr`` and ``mamba_ssm`` into
``sys.modules`` **for the duration of this test only** (the pack itself never
does this -- see REQ-8.6), so the unmodified upstream file can be imported next
to ours and compared.
"""

import importlib.util
import os
import sys
import types

import torch

from mambair.archs.mambairv2 import MambaIRv2

UPSTREAM = os.environ.get(
    "MAMBAIR_UPSTREAM_ARCH",
    r"G:/repo/comfyui/MambaIR/basicsr/archs/mambairv2_arch.py",
)

SHIMS = (
    "basicsr",
    "basicsr.archs",
    "basicsr.archs.arch_util",
    "basicsr.utils",
    "basicsr.utils.registry",
    "mamba_ssm",
    "mamba_ssm.ops",
    "mamba_ssm.ops.selective_scan_interface",
)

HP = dict(
    upscale=2, in_chans=3, img_size=64, img_range=1.0, embed_dim=24, d_state=4,
    depths=[1, 1], num_heads=[2, 2], window_size=4, inner_rank=8, num_tokens=8,
    convffn_kernel_size=5, mlp_ratio=2.0, upsampler="pixelshuffledirect",
    resi_connection="1conv",
)


def _install_shims():
    from mambair import archs as our_helpers
    from mambair import scan

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    module("basicsr").__path__ = []
    module("basicsr.archs").__path__ = []
    module(
        "basicsr.archs.arch_util",
        to_2tuple=our_helpers.to_2tuple,
        trunc_normal_=our_helpers.trunc_normal_,
    )
    module("basicsr.utils").__path__ = []

    class _Registry:
        def register(self, obj=None, **kwargs):
            def wrap(o):
                return o

            return wrap if obj is None else obj

    module("basicsr.utils.registry", ARCH_REGISTRY=_Registry())
    module("mamba_ssm").__path__ = []
    module("mamba_ssm.ops").__path__ = []
    module(
        "mamba_ssm.ops.selective_scan_interface",
        selective_scan_fn=scan.selective_scan_fn,
        selective_scan_ref=scan.selective_scan_ref,
    )


def _load_upstream():
    spec = importlib.util.spec_from_file_location("upstream_mambairv2", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    sys.modules["upstream_mambairv2"] = module
    spec.loader.exec_module(module)
    return module


def _one_hot(logits, **_kwargs):
    index = logits.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, index, 1.0)


def _with_upstream(fn):
    if not os.path.isfile(UPSTREAM):
        return "skip"
    import mambair.archs.mambairv2 as ours

    _install_shims()
    restore = ours.gumbel_hard
    try:
        upstream = _load_upstream()
        torch.nn.functional.gumbel_softmax = _one_hot
        ours.gumbel_hard = _one_hot
        return fn(upstream)
    finally:
        ours.gumbel_hard = restore
        for name in SHIMS + ("upstream_mambairv2",):
            sys.modules.pop(name, None)


def test_key_sets_are_identical():
    def check(upstream):
        ours = MambaIRv2(**HP)
        theirs = upstream.MambaIRv2(**HP)
        ours_keys, theirs_keys = set(ours.state_dict()), set(theirs.state_dict())
        assert ours_keys == theirs_keys, (
            sorted(ours_keys - theirs_keys)[:10],
            sorted(theirs_keys - ours_keys)[:10],
        )

    return _with_upstream(check)


def test_outputs_match_upstream():
    def check(upstream):
        torch.manual_seed(0)
        ours = MambaIRv2(**HP).eval()
        theirs = upstream.MambaIRv2(**HP).eval()
        theirs.load_state_dict(ours.state_dict(), strict=True)
        x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
        with torch.inference_mode():
            a = ours(x)
            b = theirs(x)
        assert a.shape == b.shape
        diff = (a - b).abs().max().item()
        assert diff < 1e-6, diff

    return _with_upstream(check)


UPSTREAM_V1 = os.environ.get(
    "MAMBAIR_UPSTREAM_V1_ARCH",
    r"G:/repo/comfyui/MambaIR/basicsr/archs/mambair_arch.py",
)

# NOTE: embed_dim must be large enough for light-SR's ChannelAttention
# (hidden_dim // squeeze_factor(30) must be >= 1), hence 60 rather than 16.
HP_V1 = dict(
    upscale=2, in_chans=3, img_size=64, img_range=1.0, embed_dim=60, d_state=4,
    depths=[1, 1], mlp_ratio=1.2, upsampler="pixelshuffledirect", resi_connection="1conv",
)


def _with_upstream_v1(fn):
    if not os.path.isfile(UPSTREAM_V1):
        return "skip"
    from mambair.archs.mambair import MambaIR as OurV1

    _install_shims()
    try:
        spec = importlib.util.spec_from_file_location("upstream_mambair_v1", UPSTREAM_V1)
        upstream = importlib.util.module_from_spec(spec)
        sys.modules["upstream_mambair_v1"] = upstream
        spec.loader.exec_module(upstream)
        return fn(upstream.MambaIR, OurV1)
    finally:
        for name in SHIMS + ("upstream_mambair_v1",):
            sys.modules.pop(name, None)


def test_v1_key_sets_are_identical():
    def check(theirs_cls, ours_cls):
        ours = ours_cls(**HP_V1)
        theirs = theirs_cls(**HP_V1)
        ours_keys, theirs_keys = set(ours.state_dict()), set(theirs.state_dict())
        assert ours_keys == theirs_keys, (
            sorted(ours_keys - theirs_keys)[:8],
            sorted(theirs_keys - ours_keys)[:8],
        )

    return _with_upstream_v1(check)


def test_v1_outputs_match_upstream():
    def check(theirs_cls, ours_cls):
        torch.manual_seed(0)
        ours = ours_cls(**HP_V1).eval()
        theirs = theirs_cls(**HP_V1).eval()
        theirs.load_state_dict(ours.state_dict(), strict=True)
        x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(2))
        with torch.inference_mode():
            a = ours(x)
            b = theirs(x)
        assert a.shape == b.shape
        diff = (a - b).abs().max().item()
        assert diff < 1e-6, diff

    return _with_upstream_v1(check)
