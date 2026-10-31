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

"""T6.1: seeded property sweeps.

No `hypothesis` dependency (NFR-3): a fixed-seed sweep over shapes and settings, sized
to run in seconds. Each case asserts a *property*, not a specific example.
"""

import random

import torch
import torch.nn.functional as F

from mambair import scan, tiling


def _case_stream(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        yield rng


def test_property_scan_equals_reference_for_random_shapes():
    """P-SCAN-1: for any valid shape and parameters, the runtime scan equals the definition."""
    rng = random.Random(1234)
    for _ in range(120):
        b = rng.randint(1, 2)
        d = rng.randint(1, 17)
        l = rng.randint(1, 600)
        n = rng.randint(1, 16)
        g = rng.choice([1, 1, 1, d]) if d > 0 else 1
        gen = torch.Generator().manual_seed(rng.randint(0, 10 ** 6))
        u = torch.randn(b, d, l, generator=gen)
        delta = torch.rand(b, d, l, generator=gen) * 0.5
        A = -torch.exp(torch.randn(d, n, generator=gen))
        B = torch.randn(b, g, n, l, generator=gen)
        C = torch.randn(b, g, n, l, generator=gen)
        softplus = rng.random() < 0.8
        ref = scan.selective_scan_ref(u, delta, A, B, C, delta_softplus=softplus)
        got = scan._scan_torch(u, delta, A, B, C, delta_softplus=softplus)
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-3)
        assert rel < 1e-4, (b, d, l, n, g, softplus, rel)


def test_property_output_size_for_random_geometries():
    """P-SIZE-1: tiled output is exactly H*scale x W*scale, whatever the tiling."""
    rng = random.Random(99)
    for _ in range(60):
        h, w = rng.randint(1, 200), rng.randint(1, 200)
        scale = rng.choice([1, 2, 3, 4])
        tile = rng.choice([16, 24, 32, 48, 64])
        overlap = rng.choice([8, 16]) if tile > 16 else 8
        x = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(rng.randint(0, 10 ** 6)))
        out = tiling.tiled_forward(
            lambda t, s=scale: F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False),
            x, scale, tile, overlap,
        )
        assert out.shape == (1, 3, h * scale, w * scale), (h, w, scale, tile, overlap, out.shape)


def test_property_constant_image_has_no_seams():
    """P-TILE-1: a constant colour survives any tiling within 1/255."""
    rng = random.Random(7)
    for _ in range(25):
        h, w = rng.randint(40, 160), rng.randint(40, 160)
        tile = rng.choice([24, 32, 48, 64])
        overlap = 8 if tile < 32 else rng.choice([8, 16])
        value = rng.random()
        x = torch.full((1, 3, h, w), value)
        out = tiling.tiled_forward(
            lambda t: F.interpolate(t, scale_factor=2, mode="nearest"), x, 2, tile, overlap
        )
        assert (out - value).abs().max() < 1 / 255, (h, w, tile, overlap, (out - value).abs().max().item())


def test_property_batch_items_are_independent():
    """P-BATCH-1: forward(batch)[i] == forward(batch[i])."""
    from mambair import node_core

    class FakeDescriptor:
        scale = 2
        patcher = None

        def __init__(self):
            self.model = torch.nn.Identity()

        def __call__(self, x):
            return F.interpolate(x, scale_factor=2, mode="nearest")

    rng = random.Random(31)
    for _ in range(10):
        n = rng.randint(2, 4)
        h = rng.randint(8, 40)
        w = rng.randint(8, 40)
        batch = torch.rand(n, h, w, 3, generator=torch.Generator().manual_seed(rng.randint(0, 10 ** 6)))
        full, _ = node_core.run_node(FakeDescriptor(), batch, tile=16, overlap=8)
        for i in range(n):
            one, _ = node_core.run_node(FakeDescriptor(), batch[i:i + 1], tile=16, overlap=8)
            assert torch.allclose(full[i], one[0], atol=1e-6), (i, (full[i] - one[0]).abs().max().item())


def test_property_triton_matches_reference_for_random_shapes():
    """P-S1: for any accepted shape and option set, the fused tier equals the definition.

    Skips without CUDA or Triton -- the tier is optional by design (NFR-3). The sweep includes the
    option combinations the models do and do not use (``D``, ``z``, ``delta_bias``, both softplus
    settings) because they select different kernel branches.
    """
    if not torch.cuda.is_available():
        return "skip"
    try:
        from mambair.scan_triton import selective_scan_triton as fused
    except Exception:  # noqa: BLE001 - triton absent or unusable
        return "skip"
    rng = random.Random(20251026)
    with torch.inference_mode():
        for _ in range(60):
            b = rng.randint(1, 2)
            d = rng.randint(1, 17)
            l = rng.randint(1, 600)
            n = rng.randint(1, 8)
            g = rng.choice([1, d])
            gen = torch.Generator(device="cuda").manual_seed(rng.randint(0, 10 ** 6))
            u = torch.randn(b, d, l, generator=gen, device="cuda") * 3
            delta = torch.rand(b, d, l, generator=gen, device="cuda") * 0.5
            A = -torch.exp(torch.randn(d, n, generator=gen, device="cuda"))
            B = torch.randn(b, g, n, l, generator=gen, device="cuda")
            C = torch.randn(b, g, n, l, generator=gen, device="cuda")
            extra = rng.random() < 0.5
            D = torch.randn(d, generator=gen, device="cuda") if extra else None
            bias = torch.randn(d, generator=gen, device="cuda") * 0.1 if extra else None
            z = torch.randn(b, d, l, generator=gen, device="cuda") if extra else None
            softplus = rng.random() < 0.8
            ref = scan.selective_scan_ref(
                u, delta, A, B, C, D=D, z=z, delta_bias=bias, delta_softplus=softplus
            )
            got = fused(u, delta, A, B, C, D=D, z=z, delta_bias=bias, delta_softplus=softplus)
            rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-3)
            assert rel < 1e-4, (b, d, l, n, g, extra, softplus, rel)
