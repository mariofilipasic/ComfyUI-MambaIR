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


"""The one node this pack adds: ``MambaIR Upscale Image``.

The primary workflow needs no node at all -- native ``Load Upscale Model`` plus
``Upscale Image (using Model)`` is enough once the architectures are registered
(see ``__init__.py``). This node exists for the knobs the native node does not
expose (tile, overlap, seed, precision) and to bound tiling deliberately.

V3-first, with a V1 wrapper over the same pure-python core
(``mambair/node_core.py``), so a pre-v0.3.48 ComfyUI still gets a working node.
"""

from __future__ import annotations

import logging

from .mambair.node_core import (
    CATEGORY,
    DESCRIPTION,
    DISPLAY_NAME,
    NODE_ID,
    OVERLAP_TOOLTIP,
    PRECISION_TOOLTIP,
    SEED_TOOLTIP,
    TILE_TOOLTIP,
    VERBOSE_TOOLTIP,
    fingerprint,
    run_node,
    validate,
)

log = logging.getLogger("MambaIR")

try:  # pragma: no cover - exercised only inside ComfyUI
    from comfy_api.latest import ComfyExtension, io

    _HAS_V3 = True
except Exception:  # noqa: BLE001 - legacy ComfyUI, or no ComfyUI at all
    ComfyExtension = None
    io = None
    _HAS_V3 = False


if _HAS_V3:

    class MambaIRUpscaleImage(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=DESCRIPTION,
                search_aliases=["mamba", "ssm", "upscale", "super resolution", "restore"],
                inputs=[
                    io.UpscaleModel.Input("upscale_model"),
                    io.Image.Input("image"),
                    io.Int.Input("tile", default=0, min=0, max=2048, step=128, tooltip=TILE_TOOLTIP),
                    io.Int.Input("overlap", default=32, min=16, max=512, step=8, tooltip=OVERLAP_TOOLTIP),
                    io.Int.Input("seed", default=0, min=-1, max=0xFFFFFFFF, tooltip=SEED_TOOLTIP),
                    io.Combo.Input("precision", options=["fp32", "bf16"], default="fp32",
                                   tooltip=PRECISION_TOOLTIP),
                    io.Boolean.Input("verbose", default=False, tooltip=VERBOSE_TOOLTIP),
                ],
                outputs=[
                    io.Image.Output(),
                    io.String.Output("status"),
                ],
            )

        @classmethod
        def validate_inputs(cls, **kwargs):
            return validate(**kwargs)

        @classmethod
        def fingerprint_inputs(cls, **kwargs):
            return fingerprint(**kwargs)

        @classmethod
        def execute(cls, upscale_model, image, tile=0, overlap=32, seed=0, precision="fp32",
                    verbose=False):
            out, status = run_node(upscale_model, image, tile, overlap, seed, precision, verbose)
            return io.NodeOutput(out, status)

    class MambaIRExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [MambaIRUpscaleImage]

    async def comfy_entrypoint():
        return MambaIRExtension()


class MambaIRUpscaleImageV1:
    """Legacy-API twin, for ComfyUI older than v0.3.48."""

    CATEGORY = CATEGORY
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    DESCRIPTION = DESCRIPTION

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "tile": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 128, "tooltip": TILE_TOOLTIP}),
                "overlap": ("INT", {"default": 32, "min": 16, "max": 512, "step": 8, "tooltip": OVERLAP_TOOLTIP}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xFFFFFFFF, "tooltip": SEED_TOOLTIP}),
                "precision": (["fp32", "bf16"], {"default": "fp32", "tooltip": PRECISION_TOOLTIP}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": VERBOSE_TOOLTIP}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return fingerprint(**kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return validate(**kwargs)

    def execute(self, upscale_model, image, tile=0, overlap=32, seed=0, precision="fp32",
                verbose=False):
        out, status = run_node(upscale_model, image, tile, overlap, seed, precision, verbose)
        return (out, status)


NODE_CLASS_MAPPINGS = {NODE_ID: MambaIRUpscaleImageV1}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: DISPLAY_NAME}
