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


"""P-SCAN-1/2/3: every scan tier equals the sequential definition."""

import torch

from mambair import scan


def _shapes():
    return [
        dict(b=2, d=8, l=1, n=4),
        dict(b=1, d=4, l=7, n=3),
        dict(b=2, d=16, l=64, n=8),
        dict(b=1, d=32, l=300, n=16),  # crosses the 256-token chunk boundary
    ]


def _inputs(b, d, l, n, groups=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(b, d, l, generator=g)
    delta = torch.rand(b, d, l, generator=g) * 0.5
    A = -torch.exp(torch.randn(d, n, generator=g))
    if groups:
        B = torch.randn(b, groups, n, l, generator=g)
        C = torch.randn(b, groups, n, l, generator=g)
    else:
        B = torch.randn(b, n, l, generator=g)
        C = torch.randn(b, n, l, generator=g)
    return u, delta, A, B, C


def test_scan_torch_matches_reference():
    for cfg in _shapes():
        u, delta, A, B, C = _inputs(**cfg)
        ref = scan.selective_scan_ref(u, delta, A, B, C, delta_softplus=True)
        got = scan._scan_torch(u, delta, A, B, C, delta_softplus=True)
        assert got.shape == u.shape, (cfg, got.shape)
        assert torch.allclose(got, ref, atol=1e-4), (cfg, (got - ref).abs().max().item())


def test_scan_optional_arguments():
    u, delta, A, B, C = _inputs(2, 16, 100, 16)
    D = torch.randn(16, generator=torch.Generator().manual_seed(1))
    bias = torch.randn(16, generator=torch.Generator().manual_seed(2))
    z = torch.randn_like(u)

    neutral = scan.selective_scan_ref(u, delta, A, B, C)
    assert torch.allclose(neutral, scan._scan_torch(u, delta, A, B, C), atol=1e-4)

    args = dict(D=D, delta_bias=bias, z=z, delta_softplus=True)
    ref = scan.selective_scan_ref(u, delta, A, B, C, **args)
    got = scan._scan_torch(u, delta, A, B, C, **args)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max().item()
    assert not torch.allclose(ref, neutral), "arguments had no effect"


def test_scan_grouped_bc():
    u, delta, A, B, C = _inputs(2, 8, 40, 4, groups=2)
    ref = scan.selective_scan_ref(u, delta, A, B, C)
    got = scan._scan_torch(u, delta, A, B, C)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max().item()


def test_scan_chunk_size_invariance():
    u, delta, A, B, C = _inputs(1, 16, 513, 8, seed=7)
    base = scan.selective_scan_ref(u, delta, A, B, C, delta_softplus=True)
    for chunk in (1, 7, 64, 256, 1024):
        got = scan._scan_torch(u, delta, A, B, C, delta_softplus=True, chunk=chunk)
        assert torch.allclose(got, base, atol=1e-4), (chunk, (got - base).abs().max().item())


def test_scan_last_state():
    u, delta, A, B, C = _inputs(2, 8, 64, 4)
    y, last = scan.selective_scan_ref(u, delta, A, B, C, return_last_state=True)
    y2, last2 = scan._scan_torch(u, delta, A, B, C, return_last_state=True)
    assert torch.allclose(y, y2, atol=1e-4)
    assert last.shape == (2, 8, 4) and torch.allclose(last, last2, atol=1e-4)


def test_scan_dispatch_reports_backend():
    u, delta, A, B, C = _inputs(1, 4, 16, 4)
    scan.reset_backend_for_tests()
    y = scan.selective_scan_fn(u, delta, A, B, C)
    assert y.shape == u.shape
    assert scan.backend_name() in ("torch", "triton", "mamba_ssm")


def _materialised_scan(u, delta, A, B, C, **kw):
    """T5.3a reference: discretise for the whole sequence, then scan.

    This is exactly the pre-T5.3a runtime path (`_prep` up front + doubling),
    rebuilt here so the test can prove that moving the discretisation inside the
    chunk loop changed nothing.
    """
    delta_bias = kw.pop("delta_bias", None)
    delta_softplus = kw.pop("delta_softplus", False)
    chunk = kw.pop("chunk", scan.CHUNK)
    u, dA, dBu, Cp = scan._prep(u, delta, A, B, C, delta_bias, delta_softplus)
    b, d, l, n = dA.shape
    y = torch.empty(b, d, l, dtype=torch.float32, device=u.device)
    carry = torch.zeros(b, d, n, dtype=torch.float32, device=u.device)
    for start in range(0, l, chunk):
        stop = min(start + chunk, l)
        a = dA[:, :, start:stop].clone()
        q = dBu[:, :, start:stop].clone()
        carry = scan._doubling(a, q, carry)
        y[:, :, start:stop] = (q * Cp[:, :, start:stop]).sum(-1)
    y = scan._finish(y, u, kw.pop("D", None), kw.pop("z", None))
    return (y, carry) if kw.pop("return_last_state", False) else y


def test_chunk_local_prep_is_bit_identical():
    """Moving the discretisation into the chunk loop must not change the result.

    Bit-exact for the ungrouped case (what MambaIR actually uses: ``B``/``C`` are
    ``(b, 1, n, l)``). With real channel *groups*, ``C`` is now broadcast instead
    of materialised, so the final reduction runs over a different stride pattern
    and may differ by one ULP -- asserted separately below, at 1e-6.
    """
    cases = [
        dict(cfg=dict(b=1, d=8, l=1, n=4), kw={}),
        dict(cfg=dict(b=2, d=16, l=7, n=8), kw=dict(delta_softplus=True)),
        dict(cfg=dict(b=1, d=32, l=300, n=16), kw=dict(delta_softplus=True)),
        dict(cfg=dict(b=1, d=4, l=300, n=8, groups=2), kw=dict(delta_softplus=True)),
        dict(cfg=dict(b=2, d=8, l=513, n=4), kw=dict(delta_softplus=True)),
    ]
    for case in cases:
        u, delta, A, B, C = _inputs(**case["cfg"], seed=11)
        kw = dict(case["kw"])
        if kw.get("delta_softplus"):
            kw["delta_bias"] = torch.randn(case["cfg"]["d"], generator=torch.Generator().manual_seed(3))
        expected = _materialised_scan(u, delta, A, B, C, **kw)
        got = scan._scan_torch(u, delta, A, B, C, **kw)
        if case["cfg"].get("groups"):
            assert torch.allclose(got, expected, rtol=0, atol=1e-6), (
                case, (got - expected).abs().max().item()
            )
        else:
            assert torch.equal(got, expected), (
                case, (got - expected).abs().max().item()
            )


def test_scan_peak_memory_does_not_scale_with_sequence_length():
    """The (b,d,L,n) intermediates must not exist any more.

    Full materialisation at these shapes would be 3 * d * L * n * 4 bytes; the
    chunk-local path should stay an order of magnitude below that.
    """
    if not torch.cuda.is_available():
        return "skip"
    b, d, l, n = 1, 64, 4096, 16
    g = torch.Generator(device="cuda").manual_seed(0)
    u = torch.randn(b, d, l, generator=g, device="cuda")
    delta = torch.rand(b, d, l, generator=g, device="cuda") * 0.5
    A = -torch.exp(torch.randn(d, n, generator=g, device="cuda"))
    B = torch.randn(b, 1, n, l, generator=g, device="cuda")
    C = torch.randn(b, 1, n, l, generator=g, device="cuda")
    materialised = 3 * d * l * n * 4 / 2 ** 20

    # Measure this call's own peak: the process-wide peak would include whatever
    # other GPU tests left allocated.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        scan._scan_torch(u, delta, A, B, C, delta_softplus=True)
        torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / 2 ** 20
    assert peak < materialised / 4, (
        f"peak {peak:.1f} MB is not comfortably below a full materialisation ({materialised:.1f} MB)"
    )
