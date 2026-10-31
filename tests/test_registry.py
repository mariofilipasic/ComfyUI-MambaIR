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


"""T2.5 / REQ-8.3 / REQ-8.5: the pack must degrade, never take ComfyUI down.

These cover code that only ever runs when something *else* is broken (a different
spandrel version, a missing dependency), which is exactly why it needs tests.
"""

import torch

from mambair import infer as inf
from mambair import registry, spandrel_arch


def _tiny_hp():
    sd = {
        "conv_first.weight": torch.zeros(48, 3, 3, 3),
        "conv_after_body.weight": torch.zeros(48, 48, 3, 3),
    }
    for j in range(2):
        p = f"layers.0.residual_group.layers.{j}."
        sd[p + "assm.selectiveScan.A_logs"] = torch.zeros(96, 8)
        sd[p + "assm.in_proj.0.weight"] = torch.zeros(48, 48, 1, 1)
        sd[p + "assm.embeddingB.weight"] = torch.zeros(64, 32)
        sd[p + "win_mhsa.relative_position_bias_table"] = torch.zeros(31 ** 2, 4)
        sd[p + "convffn1.sg.conv.weight"] = torch.zeros(24, 1, 5, 5)
    sd["upsample.0.weight"] = torch.zeros(3 * 4, 48, 3, 3)
    return inf.infer(sd)


def _runner(hp):
    arch = spandrel_arch._build(hp)
    return spandrel_arch.ModelRunner(arch, hp, tile_cap=64)


def test_descriptor_falls_back_when_optional_kwargs_are_rejected():
    """spandrel that predates size_requirements/tiling: retry without them."""
    import spandrel

    real = spandrel.ImageModelDescriptor
    made = {}

    class OlderDescriptor:
        def __init__(self, *, model, state_dict, architecture, purpose, tags,
                     supports_half, supports_bfloat16, scale, input_channels,
                     output_channels, call_fn=None):
            made["call_fn"] = call_fn
            made["scale"] = scale
            made["purpose"] = purpose
            self.model = model
            self.scale = scale
            self.purpose = purpose

    spandrel.ImageModelDescriptor = OlderDescriptor
    try:
        hp = _tiny_hp()
        runner = _runner(hp)
        descriptor = spandrel_arch._make_descriptor(runner.arch, {}, object(), hp, runner)
    finally:
        spandrel.ImageModelDescriptor = real

    assert isinstance(descriptor, OlderDescriptor)
    assert descriptor.scale == 2 and descriptor.purpose == "SR"
    assert made["call_fn"] is runner, "must keep call_fn so the native node uses our runner"
    assert descriptor._mambair_runner is runner


def test_descriptor_wraps_forward_when_call_fn_is_unsupported():
    """spandrel with no call_fn at all: our semantics move onto model.forward."""
    import spandrel

    real = spandrel.ImageModelDescriptor

    class OldestDescriptor:
        def __init__(self, *, model, state_dict, architecture, purpose, tags,
                     supports_half, supports_bfloat16, scale, input_channels,
                     output_channels):
            self.model = model
            self.scale = scale
            self.purpose = purpose

    spandrel.ImageModelDescriptor = OldestDescriptor
    try:
        hp = _tiny_hp()
        runner = _runner(hp)
        arch = runner.arch
        descriptor = spandrel_arch._make_descriptor(arch, {}, object(), hp, runner)
        assert getattr(arch, "_mambair_wrapped", False) is True

        x = torch.rand(1, 3, 32, 32)
        y = descriptor.model(x)
        assert y.shape == (1, 3, 64, 64), y.shape
        assert y.min() >= 0 and y.max() <= 1
    finally:
        spandrel.ImageModelDescriptor = real
        # the wrapped module is local to this test; drop the instance attributes
        descriptor.model.__dict__.pop("forward", None)
        descriptor.model.__dict__.pop("_mambair_wrapped", None)


def test_registration_survives_broken_architectures():
    """A failure to build architectures must cost a log line, not the process."""
    previous = registry._registered
    real_architectures = spandrel_arch.architectures

    def boom():
        raise RuntimeError("simulated spandrel breakage")

    registry._registered = None
    spandrel_arch.architectures = boom
    try:
        assert registry.register_all() == ()
    finally:
        spandrel_arch.architectures = real_architectures
        registry._registered = previous
    assert registry.register_all() != (), "registration must recover afterwards"


def test_registration_is_idempotent():
    first = registry.register_all()
    second = registry.register_all()
    assert first == second, (first, second)
