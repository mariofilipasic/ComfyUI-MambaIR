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


"""Symbols deliberately left undocumented, with the reason (api-docs REQ-2.4).

`docs/docstring-waivers.md` explains the policy; this is the machine-readable copy the
`test_docstrings.py` gate reads. Keys are `"<module>:<qualname>"` using the dotted module name
relative to the package root (e.g. ``nodes:MambaIRUpscaleImageV1.execute``).

Scope: the gate checks module- and class-level definitions. Local closures and functions defined
inside another function are implementation details and are out of scope -- which is why the four
spandrel registration classes (created inside `architectures()`) appear nowhere here.
"""

WAIVERS = {
    # ComfyUI fixes these contracts; a docstring would only restate the framework.
    "nodes:MambaIRUpscaleImage": "node class; its schema is the documentation",
    "nodes:MambaIRUpscaleImage.define_schema": "framework hook",
    "nodes:MambaIRUpscaleImage.validate_inputs": "delegates to node_core.validate",
    "nodes:MambaIRUpscaleImage.fingerprint_inputs": "delegates to node_core.fingerprint",
    "nodes:MambaIRUpscaleImage.execute": "delegates to node_core.run_node",
    "nodes:MambaIRExtension": "framework extension holder",
    "nodes:MambaIRExtension.get_node_list": "framework hook",
    "nodes:comfy_entrypoint": "framework hook",
    "nodes:MambaIRUpscaleImageV1": "legacy node class; its INPUT_TYPES is the documentation",
    "nodes:MambaIRUpscaleImageV1.INPUT_TYPES": "framework hook",
    "nodes:MambaIRUpscaleImageV1.IS_CHANGED": "delegates to node_core.fingerprint",
    "nodes:MambaIRUpscaleImageV1.VALIDATE_INPUTS": "delegates to node_core.validate",
    "nodes:MambaIRUpscaleImageV1.execute": "delegates to node_core.run_node",
    # Tiny surfaces: the signature is the documentation.
    "mambair.archs:DropPath.forward": "one-line delegation",
    "mambair.archs:DropPath.__init__": "stores one attribute",
    # Triton kernels: triton.jit parses the body, so they carry comments; the module docstring
    # documents the prefix-scan design in one place.
    "mambair.scan_triton:_combine": "Triton kernel, documented by the module docstring",
    "mambair.scan_triton:_row_scan": "Triton kernel, documented by the module docstring",
    "mambair.spandrel_arch:ModelRunner.__init__": "constructor; attributes are documented on the class",
    "mambair.spandrel_arch:ModelRunner.__call__": "spandrel's call_fn adapter; see ModelRunner.run",
    "mambair.spandrel_arch:ModelRunner.extra_repr": "debug repr",
    "mambair.spandrel_arch:ModelRunner.scale": "one-line property",
}
