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

"""ComfyUI-MambaIR -- MambaIR / MambaIRv2 as first-class ComfyUI upscale models.

Import side effect, by design: the architectures are added to ``spandrel``'s
global registry, which is what makes ComfyUI's *native*

    Load Upscale Model -> Upscale Image (using Model) -> Preview Image

work with MambaIR checkpoints, unchanged. Nothing else is patched: no
``sys.modules`` entries, no monkeypatching, no global RNG writes.

Nothing in here may raise: a broken import takes the whole ComfyUI process down.
"""

from __future__ import annotations

import logging

log = logging.getLogger("MambaIR")

__version__ = "0.1.0"

try:
    from .mambair.registry import register_all

    registered = register_all()
except Exception as exc:  # noqa: BLE001 - never break ComfyUI startup
    registered = []
    log.warning("MambaIR: disabled (%s)", exc)

__all__ = ["__version__", "registered"]

try:  # V3 API (ComfyUI >= v0.3.48)
    from . import nodes as _nodes

    _HAS_V3 = _nodes._HAS_V3
except Exception as exc:  # noqa: BLE001
    _HAS_V3 = False
    log.warning("MambaIR: node module failed to import (%s)", exc)

if _HAS_V3:
    # Export *only* comfy_entrypoint: ComfyUI's loader checks NODE_CLASS_MAPPINGS
    # first and returns, so both must not be present at once.
    from .nodes import comfy_entrypoint

    __all__ += ["comfy_entrypoint"]
else:
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        __all__ += ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    except Exception as exc:  # noqa: BLE001
        log.warning("MambaIR: no node API available (%s)", exc)
