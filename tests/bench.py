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
"""Throughput table for the machine you are on.

    python tests/bench.py [--families v2|v1|v2light|unet|all] [--tiles 128,176,256,320]
    python tests/bench.py --write

Prints a Markdown table per family and, with ``--write``, splices them into README.md between the
``<!-- bench:start -->`` / ``<!-- bench:end -->`` markers so the numbers in the docs are
reproducible instead of being a commit message.

Two kinds of absolute column, so a row is interpretable without a second row beside it:

* ``ms per output Mpx`` -- wall time per output megapixel, i.e. what a user waits for.
* ``scan GSU/s`` -- effective selective-scan state-updates per second: ``rows * tokens`` summed
  over every scan call in the tile, divided by the forward time. This is the quantity the scan
  tier optimises, so it is comparable across families (v1 runs four directional scans per block)
  and across machines. Relative-only numbers hid a 10x-too-slow tier once.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus  # noqa: E402
from mambair import registry, scan  # noqa: E402

START = "<!-- bench:start -->"
END = "<!-- bench:end -->"

#: (label, checkpoint, description) -- one row per family the pack can load.
FAMILIES = (
    ("v2", corpus.V2["x4"], "MambaIRv2 Large, x4"),
    ("v2light", corpus.V2LIGHT["x2"], "MambaIRv2 Light, x2"),
    ("v1", corpus.V1["sr_x4"], "MambaIR v1, x4, 4-direction scan"),
    ("unet", corpus.UNET["realDN"], "MambaIRUNet, x1 denoise"),
)

HEADER = [
    "| cap | tokens | backend | forward | peak VRAM | scan GSU/s | ms per output Mpx |",
    "|---|---|---|---|---|---|---|",
]

FOOTNOTE = (
    "*`scan GSU/s` = scan state-updates (rows x tokens, summed over every scan call) / forward "
    "time; `--` when a model has no scan.*\n\n"
    "*`auto` is the tier a first forward actually resolves to, so it can differ from the fused "
    "tier when the probe falls back; the row says which tier ran. Rows are measured `auto` "
    "first with the allocator emptied between rows, so each row's figures belong to that row.*"
)


def _load(path, precision="fp32"):
    from spandrel import ModelLoader

    registry.register_all()
    descriptor = ModelLoader().load_from_state_dict(corpus.load(os.path.basename(path))).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptor.model.to(device)
    runner = descriptor._mambair_runner
    runner.precision = precision  # the widget's setting, exercised without ComfyUI
    return runner, device


def _instrument(runner):
    """Count scan state-updates without touching the library.

    The architectures bind ``self.selective_scan`` per block at construction time, so wrapping
    each instance attribute measures the real calls (and their real shapes) rather than what a
    module-level patch would have caught.
    """
    stats = {"calls": 0, "updates": 0}
    for module in runner.arch.modules():
        inner = getattr(module, "selective_scan", None)
        if inner is None or getattr(inner, "_bench_wrapped", False):
            continue

        def wrapper(u, *args, _inner=inner, _stats=stats, **kwargs):
            _stats["calls"] += 1
            _stats["updates"] += int(u.shape[0]) * int(u.shape[1]) * int(u.shape[2])
            return _inner(u, *args, **kwargs)

        wrapper._bench_wrapped = True
        module.selective_scan = wrapper
    return stats


def _measure(runner, stats, device, tile, backend):
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    scan.reset_backend_for_tests()
    scan.force_backend_for_tests("torch" if backend == "torch" else None)
    x = torch.rand(1, 3, tile, tile, device=device)
    runner.run(x, tile=tile, seed=0)  # warm-up + probe
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    stats["calls"] = stats["updates"] = 0
    started = time.perf_counter()
    runner.run(x, tile=tile, seed=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
    step = max(1, tile - 32)
    out_mpx = (step * step * runner.hp.upscale ** 2) / 1e6
    ms_per_mpx = elapsed / out_mpx * 1000
    gsu = stats["updates"] / elapsed / 1e9 if stats["updates"] else None
    return elapsed, peak, ms_per_mpx, gsu, scan.backend_name()


def _table(label, path, description, runner, stats, device, tiles, backends, gpu, precision):
    lines = [
        f"Measured on **{gpu}**, `{os.path.basename(path)}` ({description}), "
        f"{precision}, overlap 32, one tile per measurement "
        f"(`python tests/bench.py --families {label}`).",
        "",
        *HEADER,
    ]
    for tile in tiles:
        for backend in backends:
            try:
                elapsed, peak, ms_mpx, gsu, used = _measure(runner, stats, device, tile, backend)
            except torch.cuda.OutOfMemoryError:
                lines.append(f"| {tile} px | {tile * tile:,} | {backend} | OOM | — | — | — |")
                torch.cuda.empty_cache()
                continue
            rate = f"{gsu:.2f}" if gsu is not None else "—"
            lines.append(
                f"| {tile} px | {tile * tile:,} | {used} | {elapsed:.2f} s | "
                f"{peak:.0f} MB | {rate} | {ms_mpx:.0f} |"
            )
    return "\n".join(lines)


def _selected(args):
    if args.model:
        return [("", args.model, "explicit --model")]
    wanted = args.families.split(",") if args.families != "all" else [f[0] for f in FAMILIES]
    chosen = []
    for label, name, description in FAMILIES:
        if label in [w.strip() for w in wanted]:
            chosen.append((label, corpus.path(name), description))
    unknown = [w for w in wanted if w.strip() not in [f[0] for f in FAMILIES]]
    for name in unknown:
        print(f"unknown family: {name} (choose from {[f[0] for f in FAMILIES]})", file=sys.stderr)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="a single checkpoint, instead of a family")
    parser.add_argument(
        "--families", default="v2", help="v2,v2light,v1,unet or all (default: v2)"
    )
    parser.add_argument("--tiles", default="128,176,256,320")
    parser.add_argument("--precision", default="fp32", choices=("fp32", "bf16"),
                        help="the node's precision widget; fp32 is the default")
    parser.add_argument(
        "--backends", default="auto,torch", help="auto,torch (default) -- fast row first"
    )
    parser.add_argument("--write", action="store_true", help="splice into README.md")
    args = parser.parse_args(argv)

    tiles = [int(t) for t in args.tiles.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    tables = []
    for label, path, description in _selected(args):
        if not os.path.isfile(path):
            print(f"checkpoint not found, skipped: {path}", file=sys.stderr)
            continue
        runner, device = _load(path, args.precision)
        stats = _instrument(runner)
        tables.append(
            _table(label, path, description, runner, stats, device, tiles, backends, gpu,
                   args.precision)
        )
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not tables:
        print("no checkpoint measured", file=sys.stderr)
        return 2
    text = "\n\n".join(tables)
    print(text)
    print(f"\n{FOOTNOTE}")

    if args.write:
        readme = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
        with open(readme, "r", encoding="utf-8") as fh:
            body = fh.read()
        if START in body and END in body:
            head, rest = body.split(START, 1)
            _, tail = rest.split(END, 1)
            body = f"{head}{START}\n{text}\n\n{FOOTNOTE}\n{END}{tail}"
            with open(readme, "w", encoding="utf-8") as fh:
                fh.write(body)
            print(f"\n(wrote the table into {readme})")
        else:
            print(f"\n(markers {START} / {END} not found in README.md; table not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
