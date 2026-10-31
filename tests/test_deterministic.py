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

"""P-SEED-1/2 and P-GRAY-1."""

import torch
import torch.nn.functional as F

from mambair import deterministic as det


def test_gumbel_onehot_shape():
    logits = torch.randn(2, 16, 8, generator=torch.Generator().manual_seed(0))
    out = det.gumbel_hard(logits)
    assert out.shape == logits.shape
    assert torch.allclose(out.sum(-1), torch.ones(2, 16))
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_seeded_router_is_reproducible():
    logits = torch.randn(1, 32, 8, generator=torch.Generator().manual_seed(3))
    with det.seeded_router(123):
        a = det.gumbel_hard(logits)
    with det.seeded_router(123):
        b = det.gumbel_hard(logits)
    with det.seeded_router(124):
        c = det.gumbel_hard(logits)
    assert torch.equal(a, b), "same seed must give identical routing"
    assert not torch.equal(a, c), "a different seed should change the routing"


def test_seeded_path_does_not_advance_global_rng():
    logits = torch.randn(1, 16, 8)
    torch.manual_seed(0)
    before = torch.random.get_rng_state().clone()
    with det.seeded_router(7):
        det.gumbel_hard(logits)
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after), "seeded routing must not consume the global RNG"


def test_luma_policy_preserves_chroma():
    x = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(5))
    y = det.luma_of(x)
    assert y.shape == (1, 1, 8, 8)
    # restoring luma by a constant offset keeps every channel difference intact
    restored = y + 0.1
    out = det.recombine_luma(restored, x)
    chroma_in = x - y
    chroma_out = out - det.luma_of(out)
    assert torch.allclose(chroma_in, chroma_out, atol=1e-5), (chroma_in - chroma_out).abs().max()


def test_run_on_luma_identity_returns_input():
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(6))
    out = det.run_on_luma(lambda t: t, x)
    assert torch.allclose(out, x, atol=1e-6)
