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


"""Node metadata and the pure implementation behind ``MambaIR Upscale Image``.

Kept outside ``nodes.py`` so it can be unit-tested without ComfyUI (``nodes.py``
needs ``comfy_api`` for the V3 schema).
"""

from __future__ import annotations

import logging
import time

import torch

from .apply import apply_descriptor
from .tiling import MIN_CAP, default_tile_cap, tile_count

log = logging.getLogger("MambaIR")

NODE_ID = "MambaIRUpscaleImage"
DISPLAY_NAME = "MambaIR Upscale Image"
CATEGORY = "image/upscaling"
DESCRIPTION = (
    "Upscale or restore an image with a MambaIR (state-space) model loaded by "
    "'Load Upscale Model'. Adds tile / overlap / seed / precision control and bounds "
    "its own VRAM use. Works with any upscale model, not only MambaIR."
)

TILE_TOOLTIP = (
    "Internal tile edge in pixels. 0 = auto (320 px on CUDA, 256 on CPU); "
    f"values below {MIN_CAP} are raised to {MIN_CAP}."
)
OVERLAP_TOOLTIP = "Feathered overlap between tiles, in pixels. Minimum 16."
SEED_TOOLTIP = "Seed for MambaIRv2's routing. -1 = randomize on every run."
PRECISION_TOOLTIP = "bf16 speeds up convolutions on CUDA; the state-space scan always runs in fp32."
VERBOSE_TOOLTIP = "Log timing and backend to the console."

__all__ = [
    "NODE_ID", "DISPLAY_NAME", "CATEGORY", "DESCRIPTION",
    "TILE_TOOLTIP", "OVERLAP_TOOLTIP", "SEED_TOOLTIP", "PRECISION_TOOLTIP",
    "VERBOSE_TOOLTIP", "run_node", "fingerprint", "validate",
]


def _progress(total_steps: int):
    """ComfyUI progress bar, or ``None`` when running outside ComfyUI."""
    try:
        import comfy.utils as comfy_utils

        return comfy_utils.ProgressBar(total_steps)
    except Exception:  # noqa: BLE001 - not running inside ComfyUI
        return None


def validate(tile: int = 0, overlap: int = 32, precision: str = "fp32", **kwargs):
    """Returns True or a human-readable error (ComfyUI's validation contract)."""
    if tile and tile <= overlap:
        return f"tile ({tile}) must be larger than overlap ({overlap})"
    if precision not in ("fp32", "bf16"):
        return f"precision must be 'fp32' or 'bf16', got {precision!r}"
    return True


def fingerprint(seed: int = 0, tile: int = 0, overlap: int = 32, precision: str = "fp32",
                **kwargs):
    """Cache key. ``seed = -1`` must never hit the cache (nanosecond resolution)."""
    if seed is not None and int(seed) < 0:
        return time.time_ns()
    return (int(seed), int(tile), int(overlap), precision)


def run_node(
    upscale_model,
    image: torch.Tensor,
    tile: int = 0,
    overlap: int = 32,
    seed: int = 0,
    precision: str = "fp32",
    verbose: bool = False,
):
    """Run the node's work; shared by the V3 node and the V1 wrapper.

    Args:
        upscale_model: A ``UPSCALE_MODEL`` value from ``Load Upscale Model``.
        image: ComfyUI ``IMAGE``, ``(n, h, w, c)``.
        tile: Internal tile edge in pixels; ``0`` derives it from the device.
        overlap: Feathered overlap in pixels.
        seed: Routing seed; ``-1`` randomizes per run and bypasses the cache.
        precision: ``"fp32"`` or ``"bf16"``.
        verbose: Log timing and backend.

    Returns:
        ``(image, status)``.

    Raises:
        ValueError: If the widget combination fails :func:`validate`.
    """
    outcome = validate(tile=tile, overlap=overlap, precision=precision)
    if outcome is not True:
        raise ValueError(outcome)

    effective_seed = int(seed)
    if effective_seed < 0:
        effective_seed = time.time_ns() % 0xFFFFFFFF  # ms resolution could repeat within a run

    if tile and int(tile) < MIN_CAP:
        # Clamped, not rejected: ComfyUI validation messages are hard errors, and a tile below
        # the floor still produces a correct (slower) result. The point is not being silent.
        log.warning(
            "MambaIR: tile %d is below the %d px floor; using %d px", int(tile), MIN_CAP, MIN_CAP
        )

    out, status = apply_descriptor(
        upscale_model,
        image,
        tile=int(tile),
        overlap=int(overlap),
        seed=effective_seed,
        precision=precision,
        verbose=verbose,
        progress=_progress(_estimate_steps(image, tile, overlap)),
    )
    return out, status


def _estimate_steps(image: torch.Tensor, tile: int, overlap: int) -> int:
    """Progress total, derived from the grid the runner will actually use."""
    try:
        h, w = int(image.shape[1]), int(image.shape[2])
        cap = max(MIN_CAP, int(tile)) if tile else default_tile_cap()
        return max(1, tile_count(h, w, cap, overlap)) * int(image.shape[0])
    except Exception:  # noqa: BLE001
        return 1
