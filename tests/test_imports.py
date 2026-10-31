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


"""REQ-5.1 / REQ-8.5 / NFR-3: import without mamba_ssm, triton, timm or basicsr."""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import sys
BLOCKED = ("mamba_ssm", "triton", "timm", "basicsr")
for name in BLOCKED:
    sys.modules[name] = None  # any import of these now raises ImportError

sys.path.insert(0, {root!r})
import mambair
from mambair import scan, infer, tiling, deterministic
from mambair.archs.mambairv2 import MambaIRv2
from mambair.archs.mambairv2light import MambaIRv2Light
from mambair.spandrel_arch import ModelRunner
m = MambaIRv2(upscale=2, in_chans=3, embed_dim=8, d_state=4, depths=[1], num_heads=[2],
              window_size=4, inner_rank=4, num_tokens=4, mlp_ratio=1.,
              upsampler='pixelshuffledirect')
print('OK', sum(p.numel() for p in m.parameters()))
"""


def _run(python=sys.executable):
    return subprocess.run(
        [python, "-c", SCRIPT.format(root=ROOT)],
        capture_output=True, text=True, timeout=300,
    )


def test_archs_import_without_optional_dependencies():
    result = _run()
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.startswith("OK"), result.stdout


def test_package_bootstrap_never_raises_without_comfyui():
    """Importing the pack outside ComfyUI must degrade, not explode (REQ-8.5)."""
    script = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "mambair_pack", {os.path.join(ROOT, "__init__.py")!r},
    submodule_search_locations=[{ROOT!r}],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["mambair_pack"] = mod
spec.loader.exec_module(mod)
print("OK", getattr(mod, "registered", None))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.startswith("OK"), result.stdout


def test_aliased_package_can_load_a_real_checkpoint():
    """Regression: `_build` must import relative to the package, not absolutely.

    ComfyUI installs this pack as `custom_nodes/<dir>/`, so `mambair.archs...`
    does not exist as an importable name there.
    """
    import corpus

    name = corpus.V2["x4"]
    if not corpus.exists(name):
        return "skip"
    script = f"""
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location(
    "mambair_pack", {os.path.join(ROOT, "__init__.py")!r},
    submodule_search_locations=[{ROOT!r}],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["mambair_pack"] = mod
spec.loader.exec_module(mod)
from spandrel import ModelLoader
sd = torch.load({corpus.path(name)!r}, map_location="cpu", weights_only=False)
d = ModelLoader().load_from_state_dict(sd)
print("OK", d.scale, type(d.model).__name__)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.startswith("OK 4 MambaIRv2"), result.stdout
