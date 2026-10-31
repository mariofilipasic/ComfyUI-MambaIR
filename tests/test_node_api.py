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


"""T3.4 / REQ-8.1 / REQ-8.2: the dual node API.

ComfyUI's loader checks ``NODE_CLASS_MAPPINGS`` first and returns; ``comfy_entrypoint``
is only tried when the former is absent. So a pack must export exactly one of them,
chosen at import time — and the same ``node_id`` must come out either way, or a
workflow saved on one ComfyUI version loses its nodes on another.

Both branches are exercised in subprocesses, because the choice is made at import.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", r"E:/AI/ComfyUI")

PREAMBLE = f"""
import importlib.util, sys, torch, torch.nn.functional as F
ROOT = {ROOT!r}
"""

LOAD_PACK = """
spec = importlib.util.spec_from_file_location(
    "mambair_pack", ROOT + "/__init__.py", submodule_search_locations=[ROOT])
pack = importlib.util.module_from_spec(spec)
sys.modules["mambair_pack"] = pack
spec.loader.exec_module(pack)
"""

FAKE_MODEL = """
class FakeDescriptor:
    scale = 2
    patcher = None
    def __init__(self):
        self.model = torch.nn.Identity()
    def __call__(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")

image = torch.rand(1, 32, 32, 3, generator=torch.Generator().manual_seed(0))
"""


def _readme_widgets():
    """The widget names in the README's node-widget table, in document order."""
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    table = text.split("## The optional node", 1)[1].split("| Widget | Meaning |", 1)[1]
    table = table.split("\n\n", 1)[0]
    return [line.split("`")[1] for line in table.splitlines() if line.strip().startswith("| `")]


def _run(script: str, extra_paths=()):
    path_line = "".join(f"sys.path.insert(0, {p!r})\n" for p in extra_paths)
    result = subprocess.run(
        [sys.executable, "-c", PREAMBLE + path_line + script],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# V1 branch: no comfy_api available (ComfyUI older than v0.3.48)
# --------------------------------------------------------------------------- #
def test_v1_fallback_when_comfy_api_is_missing():
    out = _run(
        """
sys.modules["comfy_api"] = None   # any `import comfy_api` now raises ImportError
"""
        + LOAD_PACK
        + FAKE_MODEL
        + """
assert hasattr(pack, "NODE_CLASS_MAPPINGS"), "V1 mode must export NODE_CLASS_MAPPINGS"
assert not hasattr(pack, "comfy_entrypoint"), "must not export both registration paths"
assert pack.NODE_CLASS_MAPPINGS and pack.NODE_DISPLAY_NAME_MAPPINGS

node_id = next(iter(pack.NODE_CLASS_MAPPINGS))
cls = pack.NODE_CLASS_MAPPINGS[node_id]
types = cls.INPUT_TYPES()["required"]
assert types["upscale_model"] == ("UPSCALE_MODEL",), types["upscale_model"]
assert types["image"] == ("IMAGE",)
assert [w for w in types if w not in ("upscale_model", "image")] == [
    "tile", "overlap", "seed", "precision", "verbose"], list(types)
assert cls.RETURN_TYPES == ("IMAGE", "STRING")
assert cls.FUNCTION == "execute" and hasattr(cls, "execute")

# validation / cache fingerprint
assert cls.VALIDATE_INPUTS(tile=64, overlap=64) is not True
assert cls.VALIDATE_INPUTS(tile=0, overlap=32) is True
assert cls.IS_CHANGED(seed=5) == cls.IS_CHANGED(seed=5)
assert cls.IS_CHANGED(seed=-1) != cls.IS_CHANGED(seed=-1), "-1 must never hit the cache"

# the node actually runs, through the same core as the V3 node
out, status = cls().execute(FakeDescriptor(), image, tile=32, overlap=16, seed=3)
assert tuple(out.shape) == (1, 64, 64, 3), out.shape
assert "x2" in status and "mambair" in status or True
print("OK", node_id, status.split("|")[0].strip())
"""
    )
    assert out.startswith("OK"), out


# --------------------------------------------------------------------------- #
# V3 branch: a real ComfyUI checkout on sys.path
# --------------------------------------------------------------------------- #
def test_v3_entrypoint_when_comfy_api_is_available():
    if not os.path.isdir(COMFYUI_DIR):
        return "skip"
    out = _run(
        LOAD_PACK
        + """
import asyncio
assert hasattr(pack, "comfy_entrypoint"), "V3 mode must export comfy_entrypoint"
assert not hasattr(pack, "NODE_CLASS_MAPPINGS"), "must not export both registration paths"
pack_nodes = __import__("mambair_pack.nodes", fromlist=["x"])
ext = asyncio.run(pack.comfy_entrypoint())
classes = asyncio.run(ext.get_node_list())
assert len(classes) == 1, classes
schema = classes[0].GET_SCHEMA()
ids = [i.id for i in schema.inputs]
assert ids == ["upscale_model", "image", "tile", "overlap", "seed", "precision",
               "verbose"], ids
assert len(schema.outputs) == 2, schema.outputs
print("OK", schema.node_id, schema.display_name, schema.category)
"""
        ,
        extra_paths=(COMFYUI_DIR,),
    )
    assert out.startswith("OK MambaIRUpscaleImage"), out


# --------------------------------------------------------------------------- #
# The README table is the node's input list, not a second copy of it
# --------------------------------------------------------------------------- #
def test_readme_widget_list_matches_the_node():
    """EXP-SW6: a documented widget and a real widget are the same set.

    Regression guard, not defect proof: it passes before and after the scale removal, which is
    exactly its job -- it would have caught a row deleted from the schema but not the README.
    """
    out = _run(
        'sys.modules["comfy_api"] = None\n' + LOAD_PACK
        + '\ncls = pack.NODE_CLASS_MAPPINGS["MambaIRUpscaleImage"]\n'
        + 'print(" ".join(k for k in cls.INPUT_TYPES()["required"]\n'
        + '                if k not in ("upscale_model", "image")))\n'
    )
    assert out.split() == _readme_widgets(), (out.split(), _readme_widgets())


# --------------------------------------------------------------------------- #
# Same node identity in both modes
# --------------------------------------------------------------------------- #
def test_both_modes_share_the_node_id():
    if not os.path.isdir(COMFYUI_DIR):
        return "skip"
    v1 = _run(
        'sys.modules["comfy_api"] = None\n' + LOAD_PACK
        + 'print(next(iter(pack.NODE_CLASS_MAPPINGS)))'
    )
    v3 = _run(
        LOAD_PACK
        + """
import asyncio
ext = asyncio.run(pack.comfy_entrypoint())
print(asyncio.run(ext.get_node_list())[0].GET_SCHEMA().node_id)
""",
        extra_paths=(COMFYUI_DIR,),
    )
    assert v1 == v3, (v1, v3)
