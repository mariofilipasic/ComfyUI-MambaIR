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

"""Registration and startup diagnostics.

Everything here is defensive: a failure must cost a log line, never the ComfyUI
startup (a raising custom node takes the whole process down for the user).
"""

from __future__ import annotations

import importlib.util
import logging

log = logging.getLogger("MambaIR")

__all__ = ["register_all", "scan_backends_available", "EXPECTED_FILENAMES"]

EXPECTED_FILENAMES = (
    "mambairv2_classicSR_{Base,Small,Large}_x{2,3,4}.pth",
    "mambairv2_lightSR_x{2,3,4}.pth",
    "mambair_*_SR_x{2,3,4}.pth",
    "mambair_lightSR_x{2,3,4}.pth",
    "colorDN_MambaIR_level{15,25,50}.pth",
    "MambaIR_CAR_q{10,30,40}.pth",
    "mambair_realDN.pth / mambair_realSR.pth",
)

_registered: list[str] | None = None


def _module_available(name: str) -> bool:
    """True if ``name`` is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - broken/partial installs
        return False


def scan_backends_available() -> str:
    """Which scan tiers exist on this machine, without importing any of them."""
    if _module_available("mamba_ssm"):
        return "mamba_ssm kernel"
    if _module_available("triton"):
        return "triton (probed on first use)"
    return "pytorch"


def register_all() -> tuple[str, ...]:
    """Add our architectures to ``spandrel.MAIN_REGISTRY`` (idempotent).

    Returns an immutable snapshot, so a caller cannot mutate the cached result.
    """
    global _registered
    if _registered is not None:
        return tuple(_registered)

    registered: list[str] = []
    try:
        from spandrel import MAIN_REGISTRY, ArchSupport
    except Exception as exc:  # noqa: BLE001 - spandrel missing/broken
        log.warning("MambaIR: spandrel unavailable (%s); the native upscale path is disabled", exc)
        _registered = registered
        return tuple(registered)

    try:
        from .spandrel_arch import architectures

        archs = architectures()
    except Exception as exc:  # noqa: BLE001
        log.warning("MambaIR: could not build architectures (%s)", exc)
        _registered = registered
        return tuple(registered)

    for arch in archs:
        try:
            MAIN_REGISTRY.add(ArchSupport.from_architecture(arch), ignore_duplicates=True)
            registered.append(str(arch.id))
            log.info("MambaIR: registered architecture %s", arch.id)
        except Exception as exc:  # noqa: BLE001 - never abort registration
            log.warning("MambaIR: failed to register %s (%s)", arch.id, exc)

    log.info(
        "MambaIR: scan backend = %s; "
        "put checkpoints in models/upscale_models -- e.g. %s -- releases: "
        "https://github.com/csguoh/MambaIR/releases",
        scan_backends_available(),
        ", ".join(EXPECTED_FILENAMES[:2]),
    )
    _registered = registered
    return tuple(registered)
