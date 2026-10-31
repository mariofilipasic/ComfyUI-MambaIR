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
"""Where one tile's time goes, by kernel.

    python tests/profile_tile.py [--families v2,v1] [--tile 320] [--top 15]

Prints the ops that own self CUDA time in one forward, plus the total launch count, which is what
decides whether a tile is kernel-bound or launch-bound. ``--copies`` groups ``aten::copy_`` by operand
shape instead, which is what decides *which* layout copy to remove. Aggregate numbers answer "how fast is the
tile"; only a profile answers "which part of it to change", and the two have disagreed before.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus  # noqa: E402
from mambair import registry, scan  # noqa: E402

FAMILIES = (
    ("v2", corpus.V2["x4"], "MambaIRv2 Large, x4"),
    ("v2light", corpus.V2LIGHT["x2"], "MambaIRv2 Light, x2"),
    ("v1", corpus.V1["sr_x4"], "MambaIR v1, x4, 4-direction scan"),
    ("unet", corpus.UNET["realDN"], "MambaIRUNet, x1 denoise"),
)


def _load(name):
    from spandrel import ModelLoader

    registry.register_all()
    descriptor = ModelLoader().load_from_state_dict(corpus.load(name)).eval()
    device = torch.device("cuda")
    descriptor.model.to(device)
    return descriptor._mambair_runner, device


def _profile(runner, tile, top, copies=False):
    from torch.profiler import ProfilerActivity, profile

    x = torch.rand(1, 3, tile, tile, device="cuda")
    scan.reset_backend_for_tests()
    runner.run(x, tile=tile, seed=0)  # warm-up, allocator and probe settled
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 record_shapes=copies) as prof:
        runner.run(x, tile=tile, seed=0)
    if copies:
        return _copy_report(prof)
    # ``aten::op`` rows and the kernel rows they launch both report the same device time, so
    # summing every row double counts. Keep the kernel rows only.
    try:
        from torch.autograd import DeviceType

        events = [e for e in prof.key_averages()
                  if e.self_device_time_total > 0 and e.device_type == DeviceType.CUDA]
    except (ImportError, AttributeError):
        events = [e for e in prof.key_averages()
                  if e.self_device_time_total > 0 and not e.key.startswith("aten::")]
    total = sum(e.self_device_time_total for e in events)
    launches = sum(e.count for e in events)
    events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    lines = [
        f"| op | calls | self CUDA | share |",
        "|---|---|---|---|",
    ]
    for event in events[:top]:
        lines.append(
            f"| `{event.key}` | {event.count:,} | {event.self_device_time_total / 1000:.0f} ms "
            f"| {100 * event.self_device_time_total / total:.1f}% |"
        )
    lines.append("")
    lines.append(
        f"{len(events)} distinct kernels, {launches:,} launches, "
        f"{total / 1e6:.2f} s of device time."
    )
    return "\n".join(lines)


def _copy_report(prof):
    """``aten::copy_`` grouped by operand shape: which tensors the layout copies move."""
    groups = prof.key_averages(group_by_input_shape=True)
    rows = [g for g in groups if g.key == "aten::copy_" and g.self_device_time_total > 0]
    rows.sort(key=lambda g: g.self_device_time_total, reverse=True)
    total = sum(g.self_device_time_total for g in rows)
    lines = ["| copies | shape | self CUDA | share of copies |", "|---|---|---|---|"]
    for group in rows[:20]:
        shapes = "; ".join(str(s) for s in group.input_shapes if s)[:120]
        lines.append(
            f"| {group.count:,} | `{shapes}` | {group.self_device_time_total / 1000:.0f} ms "
            f"| {100 * group.self_device_time_total / total:.1f}% |"
        )
    lines.append("")
    lines.append(f"{sum(g.count for g in rows):,} copies, {total / 1e6:.2f} s of copy time.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="v2")
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--copies", action="store_true",
                        help="group aten::copy_ by operand shape instead of the op ranking")
    args = parser.parse_args(argv)

    wanted = [w.strip() for w in args.families.split(",")]
    blocks = []
    for label, name, description in FAMILIES:
        if label not in wanted or not corpus.exists(name):
            continue
        runner, device = _load(name)
        blocks.append(f"`{os.path.basename(name)}` ({description}), {args.tile} px tile.\n")
        blocks.append(_profile(runner, args.tile, args.top, args.copies))
        del runner
        torch.cuda.empty_cache()
    if not blocks:
        print("nothing profiled", file=sys.stderr)
        return 2
    print("\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
