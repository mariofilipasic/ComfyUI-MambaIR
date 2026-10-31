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


#!/usr/bin/env python
"""Golden-output capture for REG-2 (hardening spec, task H0.1).

Purpose: make "the output did not change" a *checkable* statement rather than a hope. The
capture is taken on the unmodified tree, before any fix lands; every later change must reproduce
it exactly.

    python tests/golden.py capture --device cpu     # write tests/golden.json (twice, must match)
    python tests/golden.py check   --device cpu     # compare the current tree against it

What is captured is the **node path** (`apply_descriptor`): runner + tiling + clamp + layout, i.e.
exactly what a user's image goes through. `tests/test_scan.py` and `test_tiling.py` already pin the
parts individually.

Determinism notes
-----------------
* `seed=0` installs a local generator, so gumbel routing is reproducible.
* A 32x32 input fits in one tile at both caps, so the capture does not depend on the cap policy.
* CPU is the reference (conv/scan on CPU are reproducible run to run). The capture asserts that by
  running everything twice and requiring identical hashes -- a flaky reference must be detected at
  capture time, not blamed on a later fix.
* A *different* machine may legitimately produce different hashes (oneDNN kernels, thread counts).
  That makes this a same-machine regression check, which is what REG-2 needs. Re-capture
  deliberately, never automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import corpus  # noqa: E402

#: One file per (device, input size): a CPU hash and a CUDA hash are different claims, and so
#: are the two CUDA *tiers*. Inputs below scan.PROBE_MIN_TOKENS always run on the reference tier,
#: so a 32 px capture pins that path; the 128 px capture pins the fused Triton path.
def golden_path(device: str, size: int) -> str:
    suffix = "" if device == "cpu" else f"_{device}"
    return os.path.join(HERE, f"golden{suffix}_{size}.json")


#: Input size -> the tier that size pins, forced explicitly. 32 px (1024 tokens) is below
#: PROBE_MIN_TOKENS so it runs the reference tier; 128 px (16384 tokens) exercises the fused kernel.
#: Forcing rather than letting the probe decide keeps a recorded baseline from depending on one
#: timing sample.
SIZES = {"cpu": {32: "torch"}, "cuda": {32: "torch", 128: "triton"}}

#: every local checkpoint, in a stable order
MODELS = (
    corpus.V2LIGHT["x2"],
    corpus.V2["x2"],
    corpus.V2["x3"],
    corpus.V2["x4"],
    corpus.V1["sr_x4"],
    corpus.V1["car_q10"],
    corpus.UNET["realDN"],
)

#: the subset small enough to run in the default test suite (~7 s on CPU)
FAST = (corpus.V2LIGHT["x2"], corpus.V2["x4"], corpus.UNET["realDN"])

SIZE = 32


def _descriptor(name: str, device: str):
    from spandrel import ModelLoader

    from mambair import registry

    registry.register_all()
    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    descriptor.model.to(device)
    return descriptor


def run_one(name: str, device: str = "cpu", size: int | None = None,
            tier: str | None = None) -> torch.Tensor:
    """Run one checkpoint through the node path and return the output tensor.

    ``size`` defaults to the module-level ``SIZE`` *at call time* -- a default argument would
    bind the value at definition time and silently ignore ``--size``.
    """
    size = SIZE if size is None else size
    from mambair import scan as scan_module

    tier = tier or SIZES[device].get(size, "torch")
    scan_module.reset_backend_for_tests()
    scan_module.force_backend_for_tests(tier, trust=True)
    from mambair.apply import apply_descriptor

    descriptor = _descriptor(name, device)
    image = torch.rand(1, size, size, 3, generator=torch.Generator().manual_seed(0))
    try:
        out, _status = apply_descriptor(
            descriptor, image, tile=0, overlap=32, seed=0, precision="fp32"
        )
    finally:
        scan_module.reset_backend_for_tests()
    return out


def digest(tensor: torch.Tensor, tier: str | None = None) -> dict:
    data = tensor.detach().to(torch.float32).contiguous().numpy()
    record = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        # diagnostics for when a hash disagrees
        "sum": round(float(data.astype("float64").sum()), 6),
        "min": round(float(data.min()), 6),
        "max": round(float(data.max()), 6),
    }
    if tier is not None:
        record["tier"] = tier   # what the baseline claims to pin
    return record


def capture(device: str, models, path: str | None = None) -> dict:
    path = path or golden_path(device, SIZE)
    first = {}
    for name in models:
        if not corpus.exists(name):
            print(f"skip {name} (not found)")
            continue
        started = time.perf_counter()
        tier = SIZES[device].get(SIZE, "torch")
        first[name] = digest(run_one(name, device, tier=tier), tier)
        print(f"captured {name:38s} {time.perf_counter() - started:5.2f}s {first[name]['sha256'][:12]}")

    # determinism assertion: a second pass must reproduce the first byte for byte
    for name, record in first.items():
        again = digest(run_one(name, device))
        if again["sha256"] != record["sha256"]:
            raise SystemExit(
                f"non-deterministic capture for {name}: {record['sha256'][:12]} != "
                f"{again['sha256'][:12]} (sum {record['sum']} vs {again['sum']})\n"
                "A flaky reference cannot be used as a regression baseline."
            )

    payload = {
        "device": device,
        "input_size": SIZE,
        "seed": 0,
        "torch": torch.__version__,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": first,
    }
    merged = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
        if previous.get("device") == device:
            merged = previous.get("records", {})
    merged.update(first)
    payload["records"] = merged
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {len(first)} record(s) to {path} in {len(payload['records'])} total")
    return payload


def check(device: str, models, path: str | None = None) -> int:
    path = path or golden_path(device, SIZE)
    if not os.path.isfile(path):
        print(f"no golden file at {path}; run: python tests/golden.py capture --device {device}")
        return 2
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)["records"]
    worst = 0
    for name in models:
        if name not in records:
            print(f"skip {name} (no golden record)")
            continue
        got = digest(run_one(name, device))
        want = records[name]
        if got["sha256"] == want["sha256"]:
            print(f"ok   {name:38s} {got['sha256'][:12]}")
        else:
            worst = 1
            print(
                f"DIFF {name:38s} {want['sha256'][:12]} -> {got['sha256'][:12]}  "
                f"shape {want['shape']}->{got['shape']}  sum {want['sum']}->{got['sum']}"
            )
    print("all identical" if not worst else "REGRESSION: output changed")
    return worst


def main(argv=None):
    global SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["capture", "check"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--all", action="store_true", help=f"all {len(MODELS)} checkpoints (default: {len(FAST)})")
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--path", default=None)
    args = parser.parse_args(argv)

    SIZE = args.size
    models = MODELS if args.all else FAST
    if args.command == "capture":
        capture(args.device, models, args.path)
        return 0
    return check(args.device, models, args.path)


if __name__ == "__main__":
    raise SystemExit(main())
