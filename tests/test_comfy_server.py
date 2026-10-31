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

"""T6.3 / T7.3: a real ComfyUI install, headless server, API graph, then uninstall.

Opt-in, because it starts (and restarts) ComfyUI and takes a couple of minutes:

    RUN_COMFY_SERVER=1 python tests/run_tests.py -k comfy_server

What it proves, end to end, that no unit test can:

1. copying the pack into ``custom_nodes/`` is a valid install (ComfyUI's own loader
   picks it up, sees the V3 entrypoint, and exposes the node in ``/object_info``);
2. the *native* three-node graph from `prototype.jpg` executes through ``/prompt``
   and returns an image of the expected size (4x, so 128x128 from a 32x32 input);
3. removing the directory again leaves a clean startup: no ``MambaIR`` log lines,
   no node in ``/object_info``, no files left behind.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", r"E:/AI/ComfyUI")
HOST = "127.0.0.1"
PORT = int(os.environ.get("COMFYUI_TEST_PORT", "8199"))
BASE = f"http://{HOST}:{PORT}"
INSTALLED = os.path.join(COMFYUI_DIR, "custom_nodes", "ComfyUI-MambaIR")
CHECKPOINT = "mambairv2_classicSR_Large_x4.pth"


def _enabled():
    return os.environ.get("RUN_COMFY_SERVER") == "1" and os.path.isdir(COMFYUI_DIR)


def _http(path, payload=None, timeout=30):
    url = f"{BASE}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _start(log_path):
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--port", str(PORT), "--dont-print-server", "--disable-auto-launch"],
        cwd=COMFYUI_DIR, stdout=handle, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 420
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"ComfyUI exited with {proc.returncode}; see {log_path}")
        try:
            _http("/object_info", timeout=5)
            return proc, handle
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2)
    proc.kill()
    raise AssertionError(f"ComfyUI did not become ready within 420 s; see {log_path}")


def _stop(proc, handle):
    try:
        proc.terminate()
        proc.wait(timeout=60)
    except Exception:  # noqa: BLE001
        proc.kill()
    finally:
        handle.close()


def _install():
    if os.path.isdir(INSTALLED):
        shutil.rmtree(INSTALLED)
    shutil.copytree(
        ROOT, INSTALLED,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".specs", "prototype.jpg"),
    )


def _uninstall():
    shutil.rmtree(INSTALLED, ignore_errors=True)


def _node_ids(object_info):
    return set(object_info)


def _graph():
    return {
        "1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": CHECKPOINT}},
        "2": {"class_type": "EmptyImage",
              "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
    }


def _stale_scale_graph():
    """A prompt shaped like a pre-upgrade workflow: the removed ``scale`` input is still present.

    Every widget the node declares is supplied, as a saved workflow would: ``tile``, ``overlap``,
    ``seed``, ``precision`` and ``verbose`` are all required by ``INPUT_TYPES``, and only ``scale``
    is a key the node no longer declares. The value is deliberately 2 on an x4 checkpoint, so an
    honoured override would return 64×64.
    """
    graph = _graph()
    graph["3"] = {
        "class_type": "MambaIRUpscaleImage",
        "inputs": {
            "upscale_model": ["1", 0], "image": ["2", 0],
            "tile": 0, "overlap": 32, "seed": 0, "precision": "fp32", "verbose": False,
            "scale": 2,  # no longer declared by the node
        },
    }
    return graph


def _queued_image_size(graph):
    """Submit ``graph``, wait for it, and return the first previewed image's ``(w, h)``."""
    result = _http("/prompt", {"prompt": graph})
    prompt_id = result["prompt_id"]
    deadline = time.time() + 300
    history = {}
    while time.time() < deadline:
        history = _http(f"/history/{prompt_id}")
        if prompt_id in history:
            break
        time.sleep(1)
    else:
        raise AssertionError("the graph did not finish in 300 s")

    outputs = history[prompt_id]["outputs"]
    assert "4" in outputs, outputs
    images = outputs["4"]["images"]
    assert images, outputs
    filename = os.path.join(COMFYUI_DIR, "temp", images[0]["filename"])
    assert os.path.isfile(filename), filename
    from PIL import Image

    with Image.open(filename) as image:
        return image.size


def test_real_install_serves_the_node_and_runs_the_prototype_graph():
    if not _enabled():
        return "skip"
    log_one = os.path.join(ROOT, "comfyui_server_test.log")
    log_two = os.path.join(ROOT, "comfyui_uninstall_test.log")

    _install()
    proc, handle = _start(log_one)
    try:
        info = _http("/object_info")
        assert "MambaIRUpscaleImage" in info, "the node is missing from /object_info"
        node = info["MambaIRUpscaleImage"]
        assert node["category"] == "image/upscaling", node["category"]
        assert set(node["input"]["required"]) == {
            "upscale_model", "image", "tile", "overlap", "seed", "precision", "verbose",
        }, sorted(node["input"]["required"])

        assert _queued_image_size(_graph()) == (128, 128), "the native graph must still upscale x4"
        # EXP-SW4 / P-SW-3: a prompt written before the input was removed still runs, and the
        # model's scale decides the size. Diagnostics in an opt-in test, by design -- the structural
        # reason it holds is that execution.py validates only the keys the node declares.
        assert _queued_image_size(_stale_scale_graph()) == (128, 128), (
            "a stale scale input must not resize the output"
        )
    finally:
        _stop(proc, handle)

    # ---- uninstall: restart without the pack and expect a clean log ----
    _uninstall()
    proc, handle = _start(log_two)
    try:
        info = _http("/object_info")
        assert "MambaIRUpscaleImage" not in _node_ids(info), "node still present after uninstall"
    finally:
        _stop(proc, handle)

    assert not os.path.isdir(INSTALLED), "install directory survived uninstall"
    with open(log_two, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    assert "MambaIR" not in text, "MambaIR still logs after uninstall:\n" + "\n".join(
        line for line in text.splitlines() if "MambaIR" in line
    )
