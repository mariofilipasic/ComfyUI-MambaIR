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


"""REG-2: the golden-output check (hardening spec, task H0.1).

The reference is captured with `python tests/golden.py capture` on the unmodified tree. Default run
checks the fast subset on CPU (~10 s); the full seven-checkpoint sweep on both devices is opt-in:

    RUN_GOLDEN=1 python tests/run_tests.py -k golden

Amended after measurement (see design.md §4): the CUDA capture turned out to be bit-stable across
repeated runs on this machine, so the assertion is `sha256 ==` on both devices rather than the
1e-6 tolerance the spec allowed. If a driver or torch upgrade legitimately moves the bits, the
failure names the checkpoint and the fix is a deliberate re-capture -- never loosening this to a
tolerance, because a tolerance is exactly what would hide an in-place-aliasing regression.
"""

import json
import os

import corpus
import golden


def _records(device, size=32):
    path = golden.golden_path(device, size)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["records"]


def _assert_identical(device, models, size=32):
    records = _records(device, size)
    if records is None:
        return "skip"
    missing = [name for name in models if name not in records or not corpus.exists(name)]
    if missing:
        return "skip"
    drifted = []
    for name in models:
        want = records[name]
        got = golden.digest(golden.run_one(name, device, size))
        if got["sha256"] != want["sha256"]:
            drifted.append(
                f"{name}: {want['sha256'][:12]} -> {got['sha256'][:12]} "
                f"(sum {want['sum']} -> {got['sum']}, shape {want['shape']} -> {got['shape']})"
            )
    assert not drifted, f"{device} output changed:\n  " + "\n  ".join(drifted)
    return "ok"


def test_golden_fast_subset_cpu():
    """Default guard: light-v2, large-v2 and unet through the node path, bit-identical."""
    assert _assert_identical("cpu", golden.FAST) == "ok"


def test_golden_all_checkpoints_cpu():
    if os.environ.get("RUN_GOLDEN") != "1":
        return "skip"
    assert _assert_identical("cpu", golden.MODELS) == "ok"


def test_golden_all_checkpoints_cuda():
    import torch

    if os.environ.get("RUN_GOLDEN") != "1" or not torch.cuda.is_available():
        return "skip"
    assert _assert_identical("cuda", golden.MODELS) == "ok"


def test_golden_fused_triton_path_cuda():
    """128 px = 16 384 tokens clears PROBE_MIN_TOKENS, so this pins the fused kernel."""
    import torch

    if os.environ.get("RUN_GOLDEN") != "1" or not torch.cuda.is_available():
        return "skip"
    assert _assert_identical("cuda", golden.MODELS, size=128) == "ok"
    # the record declares the tier it pins, so a capture that silently ran elsewhere is visible
    records = _records("cuda", 128)
    wrong = sorted(name for name, rec in records.items() if rec.get("tier") != "triton")
    assert not wrong, f"128 px baselines not pinned to the fused kernel: {wrong}"
