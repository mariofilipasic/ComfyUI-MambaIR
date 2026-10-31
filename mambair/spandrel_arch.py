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


"""spandrel architectures for the MambaIR families.

Registering these makes ComfyUI's *native* ``Load Upscale Model`` /
``Upscale Image (using Model)`` pair work unchanged -- the loader calls
``spandrel.ModelLoader().load_from_state_dict()``, which walks ``MAIN_REGISTRY``
on every call.

Two details of spandrel's design matter here:

* ``ModelBase.__init__`` calls ``model.load_state_dict(state_dict)``, so
  ``descriptor.model`` must be the bare architecture (its keys must match the
  checkpoint). Anything we want to add around the forward pass therefore has to
  live in ``call_fn``, which spandrel invokes as ``call_fn(model, image)``.
* ``ImageModelDescriptor.__call__`` pads to ``size_requirements``, calls
  ``call_fn`` and clamps to ``[0, 1]``. ComfyUI's native node goes through
  ``__call__``, so ``call_fn`` is on the hot path for both nodes.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .deterministic import run_on_luma, seeded_router
from .infer import Hyperparams, UnsupportedUpsampler, canonicalize, detect_family, infer
from .tiling import MIN_CAP, bounded_forward, default_tile_cap

log = logging.getLogger("MambaIR")

__all__ = ["ModelRunner", "architectures", "ARCH_CLASSES"]

#: family -> (module leaf under ``mambair.archs``, class name)
_ARCH_CLASSES = {
    "MambaIRv2": ("mambairv2", "MambaIRv2"),
    "MambaIRv2Light": ("mambairv2light", "MambaIRv2Light"),
    "MambaIR": ("mambair", "MambaIR"),
    "MambaIRUNet": ("mambairunet", "MambaIRUNet"),
}


def _build(hp: Hyperparams) -> nn.Module:
    """Import the architecture lazily, *relative to this package*.

    An absolute import (``mambair.archs...``) breaks as soon as the pack is installed
    under ``custom_nodes/<dir>/``, where the real dotted name is
    ``custom_nodes.<dir>.mambair.archs...``. ``__package__`` resolves in both cases
    (see tests/test_imports.py::test_package_works_under_an_aliased_parent).
    """
    import importlib

    leaf, class_name = _ARCH_CLASSES[hp.family]
    module = importlib.import_module(f".archs.{leaf}", package=__package__)
    return getattr(module, class_name)(**hp.arch_kwargs())


class ModelRunner:
    """Call semantics for one MambaIR model: determinism, memory bound, colour policy.

    Installed as the descriptor's ``call_fn`` and reachable from the settings node
    as ``descriptor._mambair_runner``.
    """

    def __init__(
        self,
        arch: nn.Module,
        hp: Hyperparams,
        *,
        seed: int = 0,
        tile_cap: int | None = None,
        overlap: int = 32,
        precision: str = "fp32",
    ):
        self.arch = arch
        self.hp = hp
        self.seed = int(seed)
        self.tile_cap = int(tile_cap) if tile_cap else default_tile_cap()
        self.overlap = max(16, int(overlap))
        self.precision = precision

    @property
    def scale(self) -> int:
        return self.hp.upscale

    # spandrel's call_fn contract
    def __call__(self, model, image: torch.Tensor) -> torch.Tensor:
        return self.run(image, model=model)

    def run(self, x: torch.Tensor, *, tile=None, overlap=None, seed=None, precision=None,
            progress=None, model=None, clamp=True, on_oom=None):
        """Run the model with this runner's semantics. Never mutates its settings.

        Bounded by the tile cap, seeded for reproducibility, and colour-correct for 1-channel
        checkpoints. Always under ``torch.inference_mode()``: with autograd enabled, every scan
        keeps its doubling intermediates alive (see tests/test_spandrel_load.py).

        Args:
            x: Input, ``(n, c, h, w)`` float32 in ``[0, 1]``.
            tile: Tile cap in pixels; ``None`` uses the runner's.
            overlap: Feathered overlap; ``None`` uses the runner's.
            seed: Routing seed; ``None`` uses the runner's.
            precision: ``"fp32"`` or ``"bf16"``; ``None`` uses the runner's.
            progress: Optional progress sink.
            model: The module to call; spandrel passes its own handle to ``call_fn``.
            clamp: Clamp to ``[0, 1]``. ``apply_descriptor`` passes ``False`` because it clamps
                the result itself -- once per tile instead of twice (``REQ-5.2``).
            on_oom: Zero-argument callback run before each OOM retry, so the caller can ask the
                model manager to re-place the weights (``REQ-5.3``).

        Returns:
            ``(n, c, h*s, w*s)``, clamped to ``[0, 1]`` unless ``clamp=False``.

        Raises:
            ValueError: If ``x`` is not 4-dimensional.
        """
        if x.dim() != 4:
            raise ValueError(f"expected (N, C, H, W), got {tuple(x.shape)}")
        with torch.inference_mode():
            return self._run(x, tile=tile, overlap=overlap, seed=seed, precision=precision,
                             progress=progress, model=model, clamp=clamp, on_oom=on_oom)

    def _run(self, x: torch.Tensor, *, tile=None, overlap=None, seed=None, precision=None,
             progress=None, model=None, clamp=True, on_oom=None):
        """Body of :meth:`run`, already inside ``inference_mode``."""
        arch = model if model is not None else self.arch
        x = x.to(torch.float32)
        cap = max(MIN_CAP, int(tile)) if tile else self.tile_cap
        ov = max(16, int(overlap)) if overlap else self.overlap
        seed = self.seed if seed is None else int(seed)
        prec = precision or self.precision
        device = x.device

        def call(t: torch.Tensor) -> torch.Tensor:
            t = t.to(torch.float32)
            with seeded_router(seed, device=device):
                if prec == "bf16" and device.type in ("cuda", "xpu"):
                    with torch.autocast(device.type, dtype=torch.bfloat16):
                        return arch(t)
                return arch(t)

        def bounded(t: torch.Tensor) -> torch.Tensor:
            return bounded_forward(call, t, self.hp.upscale, cap, ov, progress=progress,
                                   on_oom=on_oom)

        if self.hp.in_chans == 1 and x.shape[1] >= 3:
            out = run_on_luma(bounded, x)
        else:
            out = bounded(x)
        return out.clamp_(0.0, 1.0) if clamp else out

    def extra_repr(self) -> str:
        return f"{self.hp.family} x{self.hp.upscale} cap={self.tile_cap} seed={self.seed}"


def _make_descriptor(model, state_dict, architecture, hp: Hyperparams, runner: ModelRunner):
    """Build the spandrel descriptor, degrading gracefully across spandrel API versions."""
    from spandrel import ImageModelDescriptor, ModelTiling, SizeRequirements

    full = dict(
        model=model,
        state_dict=state_dict,
        architecture=architecture,
        purpose=hp.purpose,
        tags=["mamba", "state-space"] + (["restoration"] if hp.upscale == 1 else ["super-resolution"]),
        supports_half=False,  # the scan stays fp32; advertising half invites dtype mismatches
        supports_bfloat16=False,
        scale=hp.upscale,
        input_channels=3,  # 1-channel checkpoints consume luma internally
        output_channels=3,
        size_requirements=SizeRequirements(multiple_of=1),
        tiling=ModelTiling.SUPPORTED,
        call_fn=runner,
    )
    try:
        descriptor = ImageModelDescriptor(**full)
    except TypeError as exc:  # spandrel API drift: retry with a smaller argument set
        log.warning("MambaIR: spandrel descriptor rejected optional kwargs (%s); retrying minimal", exc)
        minimal = {k: v for k, v in full.items() if k not in ("size_requirements", "tiling")}
        try:
            descriptor = ImageModelDescriptor(**minimal)
        except TypeError as exc2:
            # Older spandrel without `call_fn`: keep our semantics by wrapping the
            # model's own forward instead, so the native node still gets bounded,
            # seeded, colour-correct inference.
            log.warning(
                "MambaIR: spandrel descriptor does not accept call_fn (%s); "
                "installing the wrapper on the model's forward instead", exc2,
            )
            _install_forward_wrapper(model, runner)
            oldest = {k: v for k, v in minimal.items() if k != "call_fn"}
            descriptor = ImageModelDescriptor(**oldest)
    descriptor._mambair_runner = runner  # used by the settings node
    return descriptor


def _install_forward_wrapper(model: nn.Module, runner: "ModelRunner") -> None:
    """Give ``model(x)`` our call semantics when spandrel cannot take a call_fn."""
    if getattr(model, "_mambair_wrapped", False):
        return
    original = model.forward  # bound, so no recursion through nn.Module.__call__

    def forward(image):
        return runner(original, image)  # call_fn signature: (callable, image)

    model.forward = forward
    model._mambair_wrapped = True


def _load(state_dict, architecture, family: str):
    """Infer, build and wrap one checkpoint.

    Raises:
        UnsupportedModelError: If inference fails or the tensors do not match the inferred shape.
    """
    from spandrel import UnsupportedModelError

    sd = canonicalize(state_dict)
    try:
        hp = infer(sd, family)
    except UnsupportedUpsampler as exc:
        raise UnsupportedModelError(
            f"MambaIR {family}: this checkpoint uses the {exc.args[0]!r} upsampler, which the "
            f"{family} architecture does not implement; it would load but not upscale"
        ) from exc
    arch = _build(hp)
    try:
        arch.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        raise UnsupportedModelError(
            f"MambaIR {family}: checkpoint does not match the inferred architecture.\n"
            f"inferred: {hp}\n{exc}"
        ) from exc
    arch.eval()
    runner = ModelRunner(arch, hp)
    return _make_descriptor(arch, state_dict, architecture, hp, runner)


def _detect_v2(sd) -> bool:
    """Claim the state dict when it is a MambaIRv2 checkpoint (spandrel detect hook)."""
    return detect_family(canonicalize(sd)) == "MambaIRv2"


def _detect_v2light(sd) -> bool:
    """Claim the state dict when it is a MambaIRv2Light checkpoint (spandrel detect hook)."""
    return detect_family(canonicalize(sd)) == "MambaIRv2Light"


def _detect_v1(sd) -> bool:
    """Claim the state dict when it is a MambaIR (v1) checkpoint (spandrel detect hook)."""
    return detect_family(canonicalize(sd)) == "MambaIR"


def _detect_unet(sd) -> bool:
    """Claim the state dict when it is a MambaIRUNet checkpoint (spandrel detect hook)."""
    return detect_family(canonicalize(sd)) == "MambaIRUNet"


def architectures():
    """Instantiate the registered architectures (idempotent, cheap)."""
    from spandrel import Architecture

    class MambaIRv2Architecture(Architecture):
        def __init__(self):
            super().__init__(id="MambaIRv2", detect=_detect_v2, name="MambaIRv2")

        def load(self, state_dict):
            return _load(state_dict, self, "MambaIRv2")

    class MambaIRv2LightArchitecture(Architecture):
        def __init__(self):
            super().__init__(id="MambaIRv2Light", detect=_detect_v2light, name="MambaIRv2Light")

        def load(self, state_dict):
            return _load(state_dict, self, "MambaIRv2Light")

    class MambaIRArchitecture(Architecture):
        def __init__(self):
            super().__init__(id="MambaIR", detect=_detect_v1, name="MambaIR (v1)")

        def load(self, state_dict):
            return _load(state_dict, self, "MambaIR")

    class MambaIRUNetArchitecture(Architecture):
        def __init__(self):
            super().__init__(id="MambaIRUNet", detect=_detect_unet, name="MambaIRUNet")

        def load(self, state_dict):
            return _load(state_dict, self, "MambaIRUNet")

    return [
        MambaIRv2Architecture(),
        MambaIRv2LightArchitecture(),
        MambaIRArchitecture(),
        MambaIRUNetArchitecture(),
    ]


ARCH_CLASSES = tuple(_ARCH_CLASSES)
