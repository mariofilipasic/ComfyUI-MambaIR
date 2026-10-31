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


"""Apply an upscale/restoration model to a ComfyUI ``IMAGE`` tensor.

Pure torch (no ComfyUI imports), so it is unit-testable; the ComfyUI-specific
plumbing (progress bar, device management) is optional and guarded.

ComfyUI ``IMAGE`` layout: ``(N, H, W, C)``, float32 in ``[0, 1]``.
"""

from __future__ import annotations

import logging
import time
import weakref

import torch

from .tiling import MIN_CAP, bounded_forward, default_tile_cap

log = logging.getLogger("MambaIR")

__all__ = ["apply_descriptor", "output_device"]

#: patcher -> device, after a *successful* ``load_models_gpu``. A patcher that is unhashable or
#: not weak-referenceable (a duck-typed test double, a future API change) falls back to id().
_loaded_devices: "weakref.WeakKeyDictionary[object, torch.device]" = weakref.WeakKeyDictionary()
_loaded_devices_by_id: dict[int, tuple[object, torch.device]] = {}

#: RuntimeError texts that mean "the weights were moved out from under this forward".
_DEVICE_MISMATCH_MARKERS = (
    "expected all tensors to be on the same device",
    "expected object of device type",
    "is not on the expected device",
    "not on the same device",
    "different devices",
)


def output_device():
    """The device ComfyUI expects image results on (CPU outside ComfyUI)."""
    try:  # ComfyUI keeps IMAGE tensors on CPU/intermediate device
        import comfy.model_management as mm

        return mm.intermediate_device()
    except Exception:  # noqa: BLE001 - standalone use / tests
        return torch.device("cpu")


def _remember_loaded_device(patcher, device) -> None:
    """Record a successful placement without keeping an unhashable patcher from being cached."""
    try:
        _loaded_devices[patcher] = device
    except TypeError:  # unhashable or not weak-referenceable
        _loaded_devices_by_id[id(patcher)] = (patcher, device)


def _cached_loaded_device(patcher):
    """The cached device for ``patcher``, or ``None``. Identity is checked in the id fallback."""
    try:
        return _loaded_devices.get(patcher)
    except TypeError:
        entry = _loaded_devices_by_id.get(id(patcher))
        if entry is not None and entry[0] is patcher:
            return entry[1]
        return None


def _forget_loaded_device(descriptor) -> None:
    """Drop a cached placement so the next call re-loads through the model manager.

    Called when a forward proves the device is stale (ComfyUI evicted the model) and when an OOM
    retry needs the manager to make room. It never touches a model that has no patcher.
    """
    patcher = getattr(descriptor, "patcher", None)
    if patcher is None:
        return
    try:
        _loaded_devices.pop(patcher, None)
    except TypeError:
        pass
    _loaded_devices_by_id.pop(id(patcher), None)


def _clear_load_cache_for_tests() -> None:
    """Empty the placement cache (tests only; the pack never calls this)."""
    _loaded_devices.clear()
    _loaded_devices_by_id.clear()


def _load_to_device(descriptor, memory_required: float = 0.0, *, force: bool = False):
    """Ask ComfyUI's model manager to place the weights, returning their device or ``None``.

    The manager is called once per (patcher, device): after a successful load the resulting
    ``patcher.load_device`` is cached, so a multi-tile image does not repeat work that only needs
    doing once. ``force=True`` bypasses the cache and is the failure path -- an OOM retry or a
    device-mismatch retry -- so ComfyUI's own eviction/reload logic still gets the last word
    (``REQ-5.1``/``REQ-5.3``).

    Falls back to ``None`` (meaning "wherever the weights already are") when the descriptor has no
    patcher or ComfyUI is absent -- standalone use and tests. A failed load is never cached, so a
    transient failure is retried on the next call.
    """
    patcher = getattr(descriptor, "patcher", None)
    if patcher is None:
        return None
    if not force:
        cached = _cached_loaded_device(patcher)
        if cached is not None:
            return cached
    try:
        import comfy.model_management as mm

        mm.load_models_gpu([patcher], memory_required=memory_required, force_full_load=True)
        device = patcher.load_device
    except Exception as exc:  # noqa: BLE001 - fall back to wherever the weights are
        log.warning("MambaIR: could not load the model through model_management (%s)", exc)
        return None
    _remember_loaded_device(patcher, device)
    return device


def _is_device_mismatch(exc: BaseException) -> bool:
    """True when a forward failed because its weights are no longer on the input's device."""
    text = str(exc).lower()
    return any(marker in text for marker in _DEVICE_MISMATCH_MARKERS)


def _reload_after_oom(descriptor) -> None:
    """Best-effort manager call on the OOM path: forget the placement and let it re-place."""
    _forget_loaded_device(descriptor)
    _load_to_device(descriptor, force=True)


def apply_descriptor(
    descriptor,
    image: torch.Tensor,
    *,
    tile: int = 0,
    overlap: int = 32,
    seed: int | None = None,
    precision: str | None = None,
    verbose: bool = False,
    progress=None,
):
    """Run an upscale model over a ComfyUI image.

    Always runs under ``inference_mode``: model parameters require grad, so without it the autograd
    graph keeps every intermediate of a conv chain alive.

    Args:
        descriptor: A spandrel descriptor, or any callable that upscales.
        image: ComfyUI ``IMAGE``, ``(n, h, w, c)`` float32 in ``[0, 1]``; 3 channels, plus an
            optional alpha channel that is resampled and passed through.
        tile: Internal tile edge in pixels; ``0`` derives it from the device.
        overlap: Feathered overlap in pixels.
        seed: Seed for MambaIRv2's routing; ``None`` keeps the runner's own.
        precision: ``"fp32"`` or ``"bf16"``; bf16 only affects CUDA convolutions.
        verbose: Log the timing and backend line.
        progress: Optional progress sink.

    Returns:
        ``(image, status)``: the result as ``(n, h', w', c)`` on ComfyUI's output device, and a
        one-line human-readable summary.
    """
    with torch.inference_mode():
        return _apply_inner(
            descriptor, image, tile=tile, overlap=overlap, seed=seed,
            precision=precision, verbose=verbose, progress=progress,
        )


def _apply_inner(
    descriptor,
    image: torch.Tensor,
    *,
    tile: int = 0,
    overlap: int = 32,
    seed: int | None = None,
    precision: str | None = None,
    verbose: bool = False,
    progress=None,
):
    """Shared body of :func:`apply_descriptor` (which supplies ``inference_mode``)."""
    started = time.perf_counter()
    model = getattr(descriptor, "model", descriptor)
    declared = int(getattr(descriptor, "scale", 0) or getattr(model, "scale", 1) or 1)

    n, h, w, c = image.shape
    if c == 1:  # a grayscale IMAGE, not a 1-channel checkpoint (that path is run_on_luma)
        image = image.expand(-1, -1, -1, 3)
    elif c == 2:
        image = torch.cat([image, image[..., :1]], dim=-1)
        c = 3
    alpha = image[..., 3:4] if c >= 4 else None
    rgb = image[..., :3] if c > 3 else image

    runner = getattr(descriptor, "_mambair_runner", None)
    device = _load_to_device(descriptor) or next(
        (p.device for p in getattr(model, "parameters", lambda: [])()), torch.device("cpu")
    )

    def run_on(device):
        """One forward on ``device``; ``clamp=False`` keeps the single clamp below."""
        x = rgb.movedim(-1, -3).to(device=device, dtype=torch.float32)
        if runner is not None:
            out = runner.run(
                x,
                tile=(int(tile) if tile else None),
                overlap=int(overlap),
                seed=seed,
                precision=precision,
                progress=progress,
                clamp=False,
                on_oom=lambda: _reload_after_oom(descriptor),
            )
            return out, "mambair"
        # same ladder as the MambaIR path: an OOM halves the cap instead of raising
        cap = max(MIN_CAP, int(tile)) if tile else default_tile_cap(device)
        out = bounded_forward(
            lambda t: descriptor(t.float()), x, declared, cap, int(overlap), progress,
            on_oom=lambda: _reload_after_oom(descriptor),
        )
        return out, "generic"

    try:
        out, backend = run_on(device)
    except RuntimeError as exc:
        if not _is_device_mismatch(exc):  # the model was evicted between tiles
            raise
        _forget_loaded_device(descriptor)
        reloaded = _load_to_device(descriptor, force=True) or device
        log.warning(
            "MambaIR: the model moved from %s (ComfyUI evicted it); retrying on %s",
            device, reloaded,
        )
        out, backend = run_on(reloaded)

    # report the scale actually produced; the blend derives it per tile
    used_scale = max(1, out.shape[-1] // max(1, w))
    # One clamp and one crossing to ComfyUI's output device, in that order. Alpha is handled on
    # the output device so the cat never makes a second full-size buffer on the compute device.
    out = out.clamp_(0.0, 1.0).movedim(-3, -1)  # in place: the tensor is ours
    out = out.to(output_device())
    if alpha is not None:
        a = alpha.movedim(-1, -3).to(out.device)
        if a.shape[-2:] != out.shape[1:3]:
            a = torch.nn.functional.interpolate(
                a, size=out.shape[1:3], mode="bilinear", align_corners=False
            )
        out = torch.cat([out, a.movedim(-3, -1)], dim=-1)

    elapsed = time.perf_counter() - started
    family = runner.hp.family if runner is not None else type(model).__name__
    status = (
        f"{family} x{used_scale} via {backend} | {n}×{h}×{w} -> {out.shape[1]}×{out.shape[2]} | {elapsed:.2f}s"
    )
    if verbose:
        log.info("MambaIR: %s", status)
    return out, status
