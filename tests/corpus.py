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


"""Locating test checkpoints.

Set ``MAMBAIR_MODELS`` to override the folder (defaults to the local ComfyUI
install). Missing files make the dependent tests *skip*, never fail.
"""

from __future__ import annotations

import os

DEFAULT_DIR = os.environ.get(
    "MAMBAIR_MODELS", r"E:/AI/ComfyUI/models/upscale_models"
)

#: checkpoints we expect to be able to load today
V2 = {
    "x2": "mambairv2_classicSR_Large_x2.pth",
    "x3": "mambairv2_classicSR_Large_x3.pth",
    "x4": "mambairv2_classicSR_Large_x4.pth",
}

V2LIGHT = {
    "x2": "mambairv2_lightSR_x2.pth",
}

#: MambaIR v1 family (different architecture class)
V1 = {
    "sr_x4": "MambaIR-real.pth",        # pixelshuffle x4, colour
    "car_q10": "MambaIR_CAR_q10.pth",   # 1-channel, scale 1
}

#: MambaIRUNet (U-shaped, real-world denoising)
UNET = {"realDN": "realDN.pth"}

#: expected hyperparameters, straight from options/test/mambairv2/*.yml
V2_LARGE_X4 = dict(
    family="MambaIRv2",
    in_chans=3,
    embed_dim=174,
    upscale=4,
    upsampler="pixelshuffle",
    depths=(6,) * 9,
    num_heads=(6,) * 9,
    window_size=16,
    d_state=16,
    inner_rank=64,
    num_tokens=128,
    mlp_ratio=2.0,
    convffn_kernel_size=5,
    resi_connection="1conv",
)


def path(name: str) -> str:
    return os.path.join(DEFAULT_DIR, name)


def exists(name: str) -> bool:
    return os.path.isfile(path(name))


def load(name: str):
    import torch

    return torch.load(path(name), map_location="cpu", weights_only=False)


def synthetic_v2_state_dict():
    """A tiny MambaIRv2 state dict: enough keys for `infer`, small enough to build fast."""
    import torch

    sd = {
        "conv_first.weight": torch.zeros(24, 3, 3, 3),
        "conv_after_body.weight": torch.zeros(24, 24, 3, 3),
    }
    for layer in (0, 1):
        prefix = f"layers.{layer}.residual_group.layers.0."
        sd[prefix + "assm.selectiveScan.A_logs"] = torch.zeros(48, 4)
        sd[prefix + "assm.in_proj.0.weight"] = torch.zeros(48, 24, 1, 1)
        sd[prefix + "assm.embeddingB.weight"] = torch.zeros(8, 8)
        sd[prefix + "win_mhsa.relative_position_bias_table"] = torch.zeros(49, 2)
        sd[prefix + "convffn1.sg.conv.weight"] = torch.zeros(24, 1, 5, 5)
    sd["upsample.0.weight"] = torch.zeros(12, 24, 3, 3)
    return sd


def tiny_runner(**kwargs):
    """A tiny, randomly-initialised model behind the real `ModelRunner`.

    The synthetic state dict exists only as *evidence* for `infer` (family, widths,
    depth, scale); the module itself is freshly initialised, which is enough for
    behavioural tests and keeps the fixture cheap.
    """
    import torch

    from mambair import infer as inf
    from mambair.spandrel_arch import ModelRunner, _build

    hp = inf.infer(synthetic_v2_state_dict())
    torch.manual_seed(0)  # reproducible fixture weights
    arch = _build(hp).eval()
    kwargs.setdefault("tile_cap", 64)
    return ModelRunner(arch, hp, **kwargs)
