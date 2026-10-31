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


"""P-TILE-1, P-SIZE-1 (tiling half) and the OOM ladder."""

import torch
import torch.nn.functional as F

from mambair import tiling


def _resize(x, scale):
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)


def test_tiled_output_size_matches_scale():
    for h, w in ((16, 16), (37, 53), (64, 64), (100, 33)):
        for scale in (1, 2, 4):
            x = torch.rand(1, 3, h, w)
            out = tiling.tiled_forward(lambda t: _resize(t, scale), x, scale, 32, 16)
            assert out.shape == (1, 3, h * scale, w * scale), (h, w, scale, out.shape)


def test_constant_image_stays_constant():
    """A constant colour must survive tiling (feather weights are normalised)."""
    for tile, overlap in ((32, 16), (64, 16), (48, 32)):
        x = torch.full((1, 3, 100, 137), 0.42)
        out = tiling.tiled_forward(lambda t: _resize(t, 2), x, 2, tile, overlap)
        assert (out - 0.42).abs().max() < 1 / 255, (tile, overlap, (out - 0.42).abs().max().item())


def test_single_pass_when_image_fits():
    calls = []

    def fn(t):
        calls.append(tuple(t.shape))
        return _resize(t, 2)

    x = torch.rand(1, 3, 20, 20)
    tiling.tiled_forward(fn, x, 2, 64, 16)
    assert len(calls) == 1, calls


def test_positions_cover_edges():
    for size, tile, overlap in ((1024, 256, 32), (300, 128, 16), (100, 100, 16)):
        pos = tiling._positions(size, tile, overlap)
        assert pos[0] == 0
        assert pos[-1] + tile >= size, (size, tile, pos)
        assert all(b > a for a, b in zip(pos, pos[1:])), pos


def test_oom_ladder_halves_then_raises():
    seen = []

    def fn(t):
        seen.append(t.shape[-1])
        if t.shape[-1] > 128:
            raise torch.cuda.OutOfMemoryError("simulated")
        return _resize(t, 2)

    x = torch.rand(1, 3, 300, 300)
    out = tiling.bounded_forward(fn, x, 2, 512, 16)
    assert out.shape == (1, 3, 600, 600)
    assert seen[0] == 300  # first attempt used the 512 cap
    assert max(seen) > 256 and min(seen) <= 300

    def always_oom(t):
        raise torch.cuda.OutOfMemoryError("simulated")

    try:
        tiling.bounded_forward(always_oom, x, 2, 512, 16)
    except torch.cuda.OutOfMemoryError:
        pass
    else:
        raise AssertionError("expected the ladder to give up at the 128 px floor")


def test_default_cap_is_sane():
    assert tiling.default_tile_cap() in (256, 320), tiling.default_tile_cap()


def test_progress_bar_contract():
    """ComfyUI's ProgressBar exposes .update(n); plain callables are also accepted."""

    class WithUpdate:
        def __init__(self):
            self.calls = []

        def update(self, n):
            self.calls.append(n)

    class Callable:
        def __init__(self):
            self.calls = []

        def __call__(self, n):
            self.calls.append(n)

    x = torch.rand(1, 3, 100, 100)
    expected = len(tiling._positions(100, 32, 16)) ** 2
    assert expected > 1
    for progress in (WithUpdate(), Callable()):
        tiling.tiled_forward(lambda t: _resize(t, 2), x, 2, 32, 16, progress=progress)
        assert sum(progress.calls) == expected, (progress.calls, expected)
