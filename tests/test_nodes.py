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


"""R7: the settings node's core behaviour, without ComfyUI."""

import math

import torch
import torch.nn.functional as F

from mambair import node_core
from mambair.apply import apply_descriptor


class FakeDescriptor:
    """Duck-types the parts of a spandrel descriptor ``apply_descriptor`` uses."""

    scale = 2
    patcher = None

    def __init__(self):
        self.model = torch.nn.Identity()

    def __call__(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


def _image(n=1, h=48, w=64, c=3):
    return torch.rand(n, h, w, c, generator=torch.Generator().manual_seed(0))


def _params(fn):
    import inspect

    return inspect.signature(fn).parameters


def test_node_upscales_by_model_scale():
    out, status = node_core.run_node(FakeDescriptor(), _image(), tile=32, overlap=16)
    assert out.shape == (1, 96, 128, 3), out.shape
    assert out.min() >= 0 and out.max() <= 1
    assert "x2" in status, status


def test_node_rejects_bad_tile():
    try:
        node_core.run_node(FakeDescriptor(), _image(), tile=32, overlap=32)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected tile <= overlap to be rejected")


def test_fingerprint_invalidates_random_seed():
    # nanosecond-resolution key: must be unique per call, not merely "a float"
    assert isinstance(node_core.fingerprint(seed=-1), int)
    assert node_core.fingerprint(seed=-1) != node_core.fingerprint(seed=-1)
    assert node_core.fingerprint(seed=7) == node_core.fingerprint(seed=7)
    assert node_core.fingerprint(seed=7) != node_core.fingerprint(seed=8)


def test_fingerprint_reacts_to_settings():
    base = node_core.fingerprint(seed=0, tile=256, overlap=32, precision="fp32")
    assert base != node_core.fingerprint(seed=0, tile=128, overlap=32, precision="fp32")
    assert base != node_core.fingerprint(seed=0, tile=256, overlap=32, precision="bf16")


def test_scale_override_is_gone():
    """EXP-SW1 / EXP-SW5: the no-op `scale` input is deleted, not defaulted.

    It only ever logged a warning and the output size followed the model (SW-1), so no caller can
    legitimately pass it. Every check here fails on the pre-change tree: that is the point. The
    three checks are separate functions so that one failure does not hide the other two.
    """
    assert "scale" not in _params(node_core.run_node), list(_params(node_core.run_node))


def test_apply_descriptor_rejects_a_scale_argument():
    """EXP-SW5: the public python API carries no dead override either."""
    try:
        apply_descriptor(FakeDescriptor(), _image(), scale=2)
    except TypeError:
        pass
    else:
        raise AssertionError("apply_descriptor still accepts a scale argument")


def test_scale_is_not_in_the_cache_key():
    """EXP-SW5 / REG-SW3: the key still carries every setting that matters, and no other."""
    key = node_core.fingerprint(seed=0, tile=256, overlap=32, precision="fp32")
    assert len(key) == 4, key
    assert "scale" not in _params(node_core.fingerprint), list(_params(node_core.fingerprint))
    assert key != node_core.fingerprint(seed=0, tile=256, overlap=32, precision="bf16")


def test_alpha_is_preserved():
    img = _image(c=4)
    out, _ = node_core.run_node(FakeDescriptor(), img, tile=32, overlap=16)
    assert out.shape[-1] == 4, out.shape
    expected = F.interpolate(
        img[..., 3:4].movedim(-1, -3), size=out.shape[1:3], mode="bilinear", align_corners=False
    )[0, 0]
    assert torch.allclose(out[..., 3], expected, atol=1e-5)


def test_batch_is_processed_independently():
    img = _image(n=3, h=32, w=32)
    out, _ = node_core.run_node(FakeDescriptor(), img, tile=32, overlap=16)
    assert out.shape[0] == 3
    for i in range(3):
        single, _ = node_core.run_node(FakeDescriptor(), img[i:i + 1], tile=32, overlap=16)
        assert torch.allclose(out[i], single[0], atol=1e-6)
