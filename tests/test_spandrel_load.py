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


"""R1/R2: registration, loading a real checkpoint through spandrel, running it."""

import torch

import corpus
from mambair import registry

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_architectures_are_registered():
    ids = registry.register_all()
    from spandrel import MAIN_REGISTRY

    for arch_id in ("MambaIRv2", "MambaIRv2Light", "MambaIR", "MambaIRUNet"):
        assert arch_id in MAIN_REGISTRY, (arch_id, ids)
    assert registry.register_all() == ids, "registration must be idempotent"


def test_spandrel_loads_real_checkpoint():
    name = corpus.V2["x4"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    assert descriptor.scale == 4, descriptor.scale
    assert descriptor.purpose == "SR", descriptor.purpose
    assert descriptor.input_channels == 3

    descriptor.model.to(DEV)
    x = torch.rand(1, 3, 24, 24, device=DEV)
    with torch.inference_mode():
        y = descriptor(x)
    assert y.shape == (1, 3, 96, 96), y.shape
    assert y.min() >= 0 and y.max() <= 1
    assert not torch.isnan(y).any()


def test_wrapper_is_reproducible_with_a_fixed_seed():
    """REQ-4.3: same input + seed => identical output, even though v2 routes with Gumbel."""
    name = corpus.V2["x4"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    descriptor.model.to(DEV)
    runner = descriptor._mambair_runner
    x = torch.rand(1, 3, 16, 16, device=DEV)
    a = runner.run(x, seed=1234, tile=64)
    b = runner.run(x, seed=1234, tile=64)
    c = runner.run(x, seed=99, tile=64)
    assert torch.equal(a, b), (a - b).abs().max().item()
    assert not torch.equal(a, c), "a different seed should change the routing"


def test_scale_1_checkpoint_reports_restoration():
    """REQ-1.5 + REQ-4.5: a 1-channel restoration checkpoint rides the same path,
    consuming luma and keeping the input's chroma."""
    name = corpus.V1["car_q10"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    assert descriptor.scale == 1, descriptor.scale
    assert descriptor.purpose == "Restoration", descriptor.purpose

    descriptor.model.to(DEV)
    x = torch.rand(1, 3, 16, 16, device=DEV)
    y = descriptor._mambair_runner.run(x, seed=0, tile=64)
    assert y.shape == x.shape, y.shape

    from mambair.deterministic import luma_of

    chroma_in = x - luma_of(x)
    chroma_out = y - luma_of(y)
    # clipping can saturate a few pixels, so compare the bulk of the chroma
    err = (chroma_in - chroma_out).abs()
    assert err.median() < 2e-2, err.median().item()


def test_v1_checkpoints_load_and_run():
    name = corpus.V1["sr_x4"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    assert descriptor.scale == 4 and descriptor.purpose == "SR"
    descriptor.model.to(DEV)
    x = torch.rand(1, 3, 16, 16, device=DEV)
    y = descriptor._mambair_runner.run(x, seed=0, tile=64)
    assert y.shape == (1, 3, 64, 64), y.shape
    assert y.min() >= 0 and y.max() <= 1 and not torch.isnan(y).any()


def test_v2_light_checkpoint_loads_and_runs():
    name = corpus.V2LIGHT["x2"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    assert descriptor.scale == 2
    descriptor.model.to(DEV)
    x = torch.rand(1, 3, 16, 16, device=DEV)
    y = descriptor._mambair_runner.run(x, seed=0, tile=64)
    assert y.shape == (1, 3, 32, 32), y.shape


def test_runner_is_gradient_free():
    """Regression: without inference_mode a Large model's 54 scans keep their
    autograd graphs alive (measured 20+ GiB on a 64 px image)."""
    name = corpus.V2["x4"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    descriptor.model.to(DEV)
    runner = descriptor._mambair_runner
    seen = {}
    original = descriptor.model.forward

    def spy(x):
        seen["grad_enabled"] = torch.is_grad_enabled()
        return original(x)

    descriptor.model.forward = spy
    runner.run(torch.rand(1, 3, 16, 16, device=DEV), tile=64, seed=0)
    assert seen.get("grad_enabled") is False, "forward ran with autograd enabled"


def test_unet_checkpoint_loads_and_runs():
    """MambaIRUNet: U-shaped denoiser, scale 1, same native path."""
    name = corpus.UNET["realDN"]
    if not corpus.exists(name):
        return "skip"
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    assert descriptor.scale == 1 and descriptor.purpose == "Restoration"
    descriptor.model.to(DEV)
    x = torch.rand(1, 3, 32, 32, device=DEV)
    y = descriptor._mambair_runner.run(x, seed=0, tile=64)
    assert y.shape == x.shape, y.shape
    assert y.min() >= 0 and y.max() <= 1 and not torch.isnan(y).any()
