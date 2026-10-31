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
"""Scan cost, isolated and in context.

    python tests/bench_scan.py [--families v2,v1] [--tiles 320] [--tiers triton,torch]

Two numbers per family and tier, because they disagree and the difference is diagnostic:

* **isolated** -- the tensors a real forward actually passed to the scan, replayed on their own.
  This is the kernel's cost, and the only number a kernel change should be judged by.
* **in context** -- a full forward with the real scan, minus a full forward with the scan replaced
  by ``zeros_like``. This is the user's marginal cost. It is biased *low*: removing the scan also
  relieves cache and bandwidth pressure on the surrounding ops, which then run faster.

The call shapes and layouts come from a real forward rather than being reconstructed, so a family
whose scan geometry changes cannot silently be benched against the wrong shape. The count of calls
per forward is reported too: a per-call number is meaningless without it.
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


def _load(path):
    from spandrel import ModelLoader

    registry.register_all()
    descriptor = ModelLoader().load_from_state_dict(corpus.load(os.path.basename(path))).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptor.model.to(device)
    return descriptor._mambair_runner, device


def _spec(u, args, kwargs):
    """Shapes and options of one scan call, so it can be replayed without the tensors.

    The tensors are *not* kept: a v1 forward's scan arguments are >1 GB and holding them alive
    alongside the model changes the very thing this script measures.
    """
    delta, A, B, C = args[0], args[1], args[2], args[3]
    D = args[4] if len(args) > 4 else kwargs.get("D")
    return dict(
        u=tuple(u.shape),
        delta=tuple(delta.shape),
        A=tuple(A.shape),
        B=tuple(B.shape),
        C=tuple(C.shape),
        has_D=D is not None,
        has_z=kwargs.get("z") is not None,
        has_bias=kwargs.get("delta_bias") is not None,
        softplus=bool(kwargs.get("delta_softplus", False)),
        last=bool(kwargs.get("return_last_state", False)),
    )


def _capture_calls(runner, device, tile):
    """One real forward at the measured tile: the scan call specs and how often each repeats."""
    specs, counts = [], {}

    def counting(u, *args, **kwargs):
        spec = _spec(u, args, kwargs)
        key = tuple(spec.values())
        if key not in counts:
            specs.append(spec)
            counts[key] = 0
        counts[key] += 1
        return real(u, *args, **kwargs)

    modules = [m for m in runner.arch.modules() if getattr(m, "selective_scan", None) is not None]
    originals = [m.selective_scan for m in modules]
    real = originals[0]
    try:
        for module in modules:
            module.selective_scan = counting
        with torch.no_grad():
            runner.run(torch.rand(1, 3, tile, tile, device=device), tile=tile, seed=0)
    finally:
        for module, fn in zip(modules, originals):
            module.selective_scan = fn
    return [(spec, counts[tuple(spec.values())]) for spec in specs]


def _build(spec, device):
    """Random tensors with the recorded shapes, so the kernel sees the same geometry."""
    rows = spec["u"][1]
    return dict(
        u=torch.randn(*spec["u"], device=device),
        delta=torch.randn(*spec["delta"], device=device) * 0.1,
        A=-torch.rand(*spec["A"], device=device),
        B=torch.randn(*spec["B"], device=device),
        C=torch.randn(*spec["C"], device=device),
        D=torch.randn(rows, device=device) if spec["has_D"] else None,
        z=torch.randn(*spec["u"], device=device) if spec["has_z"] else None,
        delta_bias=(torch.randn(spec["delta"][1], device=device) * 0.1
                    if spec["has_bias"] else None),
        delta_softplus=spec["softplus"],
        return_last_state=spec["last"],
    )


def _time(fn, repeat):
    """Best-of-N milliseconds for one call, after two warm-ups."""
    fn()
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeat):
        started, ended = torch.cuda.Event(True), torch.cuda.Event(True)
        started.record()
        fn()
        ended.record()
        torch.cuda.synchronize()
        best = min(best, started.elapsed_time(ended))
    return best


def _measure_family(label, path, description, tile, tiers, repeat):
    runner, device = _load(path)
    signatures = _capture_calls(runner, device, tile)
    calls = sum(count for _, count in signatures)
    updates = sum(int(spec["u"][0]) * int(spec["u"][1]) * int(spec["u"][2]) * count
                  for spec, count in signatures)

    lines = [
        f"`{os.path.basename(path)}` ({description}), {tile} px tile: "
        f"{calls} scan calls per forward over {len(signatures)} distinct "
        f"shape{'s' if len(signatures) != 1 else ''} "
        f"({', '.join(str(spec['u']) for spec, _ in signatures)}).",
        "",
        "| tier | forward | scan share | per call isolated | isolated GSU/s "
        "| per call in context | in-context GSU/s | gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    x = torch.rand(1, 3, tile, tile, device=device)
    for tier in tiers:
        scan.reset_backend_for_tests()
        scan.force_backend_for_tests(tier, trust=True)
        isolated_ms = 0.0
        with torch.no_grad():
            for spec, count in signatures:
                kwargs = _build(spec, device)
                isolated_ms += count * _time(
                    lambda kw=kwargs: scan.selective_scan_fn(**kw), repeat
                )
            full_ms = _time(lambda: runner.run(x, tile=tile, seed=0), repeat)
            stubbed = [m for m in runner.arch.modules()
                       if getattr(m, "selective_scan", None) is not None]
            originals = [m.selective_scan for m in stubbed]
            for module in stubbed:
                module.selective_scan = lambda u, *a, **k: torch.zeros_like(u)
            try:
                stub_ms = _time(lambda: runner.run(x, tile=tile, seed=0), repeat)
            finally:
                for module, fn in zip(stubbed, originals):
                    module.selective_scan = fn

        in_context = max(full_ms - stub_ms, 0.0)
        per_context = in_context / calls if calls else float("nan")
        per_isolated = isolated_ms / calls if calls else float("nan")
        gap = per_context / per_isolated if per_isolated else float("nan")
        context_gsu = updates / (in_context / 1000) / 1e9 if in_context else float("nan")
        isolated_gsu = updates / (isolated_ms / 1000) / 1e9 if isolated_ms else float("nan")
        share = 100 * in_context / full_ms if full_ms else float("nan")
        lines.append(
            f"| {tier} | {full_ms / 1000:.2f} s | {share:.0f}% | {per_isolated:.1f} ms "
            f"| {isolated_gsu:.2f} | {per_context:.1f} ms | {context_gsu:.2f} | {gap:.2f}x |"
        )
    del runner, signatures
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="v2,v1")
    parser.add_argument("--tiles", default="320")
    parser.add_argument("--tiers", default="triton")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA device; this benchmark is CUDA-only", file=sys.stderr)
        return 2
    wanted = [w.strip() for w in args.families.split(",")]
    tiers = [t.strip() for t in args.tiers.split(",")]
    blocks = []
    for tile in [int(t) for t in args.tiles.split(",")]:
        for label, name, description in FAMILIES:
            if label not in wanted:
                continue
            if not corpus.exists(name):
                print(f"checkpoint not found, skipped: {name}", file=sys.stderr)
                continue
            blocks.append(_measure_family(label, corpus.path(name), description, tile, tiers,
                                          args.repeat))
    if not blocks:
        print("nothing measured", file=sys.stderr)
        return 2
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
