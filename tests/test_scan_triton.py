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


"""T5.1: the Triton tier and the probe that guards it.

Triton is optional. These tests skip when it is not installed or cannot compile
on this machine -- the pack's behaviour without it is covered by `test_imports`
and `test_scan_dispatch_reports_backend`.
"""

import logging

import torch

from mambair import scan


def _tier():
    if not torch.cuda.is_available():
        return None
    try:
        from mambair.scan_triton import selective_scan_triton

        return selective_scan_triton
    except Exception:  # noqa: BLE001 - triton absent or unusable
        return None


def _inputs(b, d, l, n, *, groups=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.randn(b, d, l, generator=g, device="cuda") * 3
    delta = torch.rand(b, d, l, generator=g, device="cuda") * 0.5
    A = -torch.exp(torch.randn(d, n, generator=g, device="cuda"))
    if groups:
        B = torch.randn(b, groups, n, l, generator=g, device="cuda")
        C = torch.randn(b, groups, n, l, generator=g, device="cuda")
    else:
        B = torch.randn(b, n, l, generator=g, device="cuda")
        C = torch.randn(b, n, l, generator=g, device="cuda")
    return u, delta, A, B, C


def test_triton_matches_the_reference():
    fn = _tier()
    if fn is None:
        return "skip"
    with torch.inference_mode():
        for cfg in (
            dict(b=1, d=8, l=1, n=4),
            dict(b=2, d=16, l=97, n=16),
            dict(b=1, d=32, l=1024, n=16),
            dict(b=1, d=4, l=300, n=8, groups=2),
        ):
            args = _inputs(**cfg)
            ref = scan.selective_scan_ref(*[t.double().cpu() for t in args]).float().cuda()
            got = fn(*args)
            scale = max(ref.abs().max().item(), 1e-3)
            rel = (got - ref).abs().max().item() / scale
            assert rel < 1e-4, (cfg, rel)


def test_triton_handles_optional_arguments():
    fn = _tier()
    if fn is None:
        return "skip"
    u, delta, A, B, C = _inputs(1, 16, 512, 16)
    g1 = torch.Generator(device="cuda").manual_seed(1)
    g2 = torch.Generator(device="cuda").manual_seed(2)
    D = torch.randn(16, generator=g1, device="cuda")
    bias = torch.randn(16, generator=g2, device="cuda")
    z = torch.randn_like(u)
    with torch.inference_mode():
        ref = scan.selective_scan_ref(
            u, delta, A, B, C, D=D, z=z, delta_bias=bias, delta_softplus=True
        )
        got = fn(u, delta, A, B, C, D, z, bias, True)
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-3)
    assert rel < 1e-4, rel


def test_probe_uses_a_fixed_slice_whatever_the_first_length():
    if _tier() is None:
        return "skip"
    scan.reset_backend_for_tests()
    try:
        big = _inputs(1, 8, scan.PROBE_TOKENS * 3, 16)
        with torch.inference_mode():
            scan.selective_scan_fn(*big, delta_softplus=True)
        assert scan._probe_state == "adopted"
        assert scan._triton_ok is True
    finally:
        scan.reset_backend_for_tests()


def test_probe_rejects_a_wrong_kernel():
    """A kernel that is fast but wrong must never be adopted."""
    if not torch.cuda.is_available():
        return "skip"
    wrong = lambda *a, **k: torch.zeros_like(a[0])  # noqa: E731
    scan.reset_backend_for_tests()
    # pre-populate the device cache so `_resolve` does not swap in the real kernel
    scan._resolved["cuda"] = "torch"
    scan._triton_fn = wrong
    try:
        u, delta, A, B, C = _inputs(1, 8, scan.PROBE_TOKENS, 16)
        with torch.inference_mode():
            got = scan.selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
            ref = scan._scan_torch(u, delta, A, B, C, delta_softplus=True)
        assert scan._probe_state == "rejected", scan._probe_state
        assert scan._triton_ok is False
        assert torch.allclose(got, ref, atol=1e-5), "must fall back to the PyTorch tier"
    finally:
        scan.reset_backend_for_tests()


def test_probe_ignores_the_users_first_tensor_values():
    """The verdict is permanent, so image content must not decide it.

    A plain black image through the real model produced a 2.48e-4 disagreement and permanently
    rejected the fused tier, even though that tier was closer to the double-precision oracle than
    the reference tier on the same data (design.md D-5). NaN input is the sharpest form of the
    same test: the probe must generate its own tensors, so it still adopts.
    """
    if _tier() is None:
        return "skip"
    scan.reset_backend_for_tests()
    try:
        u, delta, A, B, C = _inputs(1, 8, scan.PROBE_TOKENS, 16)
        poisoned = torch.full_like(u, float("nan"))
        with torch.inference_mode():
            scan.selective_scan_fn(poisoned, delta, A, B, C, delta_softplus=True)
        assert scan._probe_state == "adopted", scan._probe_state
        assert scan._triton_ok is True
    finally:
        scan.reset_backend_for_tests()


def test_probe_adopts_the_real_kernel():
    if _tier() is None:
        return "skip"
    scan.reset_backend_for_tests()
    try:
        # a representative length: below PROBE_MIN_TOKENS the tier is not consulted
        u, delta, A, B, C = _inputs(1, 8, scan.PROBE_TOKENS, 16)
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger = logging.getLogger("MambaIR")
        logger.addHandler(handler)
        try:
            with torch.inference_mode():
                scan.selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
        finally:
            logger.removeHandler(handler)
        # the log says *why* a rejection happened; without it a flake is undiagnosable
        reason = [r.getMessage() for r in records]
        assert scan._probe_state == "adopted", f"{scan._probe_state}: {reason}"
        assert scan._triton_ok is True, f"the fused kernel is size-stable; no threshold: {reason}"
        assert "triton" in scan.backend_name(), scan.backend_name()
    finally:
        scan.reset_backend_for_tests()


def test_triton_is_used_for_long_sequences_too():
    """The fused kernel never materialises (b,d,L,n), so it wins at every size."""
    if _tier() is None:
        return "skip"
    scan.reset_backend_for_tests()
    real = scan._triton_fn
    try:
        u, delta, A, B, C = _inputs(1, 8, scan.PROBE_TOKENS, 16)
        with torch.inference_mode():
            scan.selective_scan_fn(u, delta, A, B, C, delta_softplus=True)  # probe adopts

        calls = {"n": 0}
        real = scan._triton_fn

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        scan._triton_fn = counting
        for length in (scan.PROBE_TOKENS, 65536):
            args = _inputs(1, 8, length, 16)
            with torch.inference_mode():
                scan.selective_scan_fn(*args, delta_softplus=True)
        assert calls["n"] == 2, f"Triton should serve every length, got {calls['n']} calls"
    finally:
        scan._triton_fn = real
        scan.reset_backend_for_tests()


def test_row_scan_handles_short_and_ragged_lengths():
    """EXP-S2: the block loop masks its tail, so ``L < BT`` and ``L % BT != 0`` must still agree.

    ``L = 8`` is also the probe's JIT warm-up length, and ``L = 1`` the degenerate case a single
    token tile can produce.
    """
    fn = _tier()
    if fn is None:
        return "skip"
    with torch.inference_mode():
        for length in (1, 8, 31, 33, 97):
            args = _inputs(1, 8, length, 16)
            ref = scan.selective_scan_ref(*[t.double().cpu() for t in args]).float().cuda()
            got = fn(*args)
            rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-3)
            assert rel < 1e-4, (length, rel)
        args = _inputs(1, 6, 301, 8, groups=6)  # groups == channels, ragged tail
        ref = scan.selective_scan_ref(*[t.double().cpu() for t in args]).float().cuda()
        got = fn(*args)
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-3)
        assert rel < 1e-4, rel


def test_row_scan_returns_the_last_state():
    """EXP-S2: the carry leaves in a register, so ``return_last_state`` must still be exact."""
    fn = _tier()
    if fn is None:
        return "skip"
    with torch.inference_mode():
        for cfg in (dict(b=1, d=8, l=1000, n=16), dict(b=2, d=16, l=47, n=8)):
            u, delta, A, B, C = _inputs(**cfg)
            got_y, got_state = fn(u, delta, A, B, C, delta_softplus=True, return_last_state=True)
            ref_y, ref_state = scan._scan_torch(
                u, delta, A, B, C, delta_softplus=True, return_last_state=True
            )
            for got, ref, what in ((got_y, ref_y, "y"), (got_state, ref_state, "state")):
                rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-3)
                assert rel < 1e-4, (cfg, what, rel)


def test_fused_tier_has_no_carry_stage():
    """EXP-S7: the chunk-carry surface is gone -- one kernel serves the whole sequence."""
    try:
        import mambair.scan_triton as module
    except Exception:  # noqa: BLE001 - triton absent or unusable
        return "skip"
    for gone in ("CHUNK", "_chunk_carry", "_carry_scan", "_chunk_apply"):
        assert not hasattr(module, gone), f"{gone} is dead tuning surface (EXP-S7)"
    assert callable(module.selective_scan_triton)


def test_fused_tier_allocates_only_its_output():
    """EXP-S3: peak is the output buffer plus the state vector, with no per-chunk carry arrays.

    Pre-change the three-pass kernel held ``3 * rows * nchunks * d_state`` on top of the output
    (1.37x at these shapes), which is what this bound excludes.
    """
    fn = _tier()
    if fn is None:
        return "skip"
    d, length, groups = 1440, 32768, 4
    u, delta, A, B, C = _inputs(1, d, length, 16, groups=groups)
    args = (u, delta, A, B, C, torch.ones(d, device="cuda"))
    with torch.inference_mode():
        fn(*args, delta_softplus=True)  # warm: the JIT compile must not land in the measurement
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn(*args, delta_softplus=True)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    output = d * length * 4
    assert peak <= 1.2 * output, (
        f"peak {peak / 2 ** 20:.0f} MB for a {output / 2 ** 20:.0f} MB output: "
        "a per-chunk carry array came back"
    )


def test_cpu_never_imports_triton():
    previous = scan._import_triton
    scan._import_triton = lambda: (_ for _ in ()).throw(AssertionError("triton imported on CPU"))
    scan.reset_backend_for_tests()
    try:
        u = torch.randn(1, 4, 32)
        delta = torch.rand(1, 4, 32) * 0.5
        A = -torch.exp(torch.randn(4, 4))
        B = torch.randn(1, 4, 32)
        C = torch.randn(1, 4, 32)
        y = scan.selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
        assert y.shape == u.shape
        assert scan.backend_name() == "torch"
    finally:
        scan._import_triton = previous
        scan.reset_backend_for_tests()
