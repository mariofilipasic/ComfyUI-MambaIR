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

"""REQ-7.3: the docstrings survive being rendered by the standard library.

`pydoc` is in the standard library, so this costs nothing in the dependency-free test path and
catches the failure mode that matters: a section written in a way that only looks structured.
"""

import pydoc

import corpus  # noqa: F401  (ensures tests/ is on the path for the runner and pytest alike)


TIER1 = (
    ("mambair.scan", "selective_scan_fn"),
    ("mambair.scan", "selective_scan_ref"),
    ("mambair.scan_triton", "selective_scan_triton"),
    ("mambair.infer", "Hyperparams"),
    ("mambair.infer", "infer"),
    ("mambair.infer", "detect_family"),
    ("mambair.tiling", "tiled_forward"),
    ("mambair.tiling", "bounded_forward"),
    ("mambair.apply", "apply_descriptor"),
    ("mambair.spandrel_arch", "ModelRunner.run"),
    ("mambair.node_core", "run_node"),
)


def test_pydoc_renders_the_sections():
    import importlib

    problems = []
    for module_name, name in TIER1:
        module = importlib.import_module(module_name)
        obj = module
        for part in name.split("."):
            obj = getattr(obj, part)
        rendered = pydoc.render_doc(obj)
        section = "Attributes:" if name == "Hyperparams" else "Args:"
        if section not in rendered:
            problems.append(f"{module_name}.{name}: {section} missing from the rendered help")
        if "Returns:" not in rendered and name != "Hyperparams":
            problems.append(f"{module_name}.{name}: Returns: missing from the rendered help")
    assert not problems, "\n".join(problems)


def test_module_help_is_browsable():
    """`python -m pydoc mambair.scan` must produce structure, not one runaway paragraph."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pydoc", "mambair.scan"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "Args:" in out.stdout and "Returns:" in out.stdout
    assert len(out.stdout.splitlines()) > 40, "the rendered module page looks truncated"
