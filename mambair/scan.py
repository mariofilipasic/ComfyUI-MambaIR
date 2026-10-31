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


"""The selective scan, without a CUDA toolchain.

MambaIR's hot loop is one first-order linear recurrence per
``(batch, channel, state)`` triple::

    x_t = a_t * x_{t-1} + b_t          y_t = <x_t, C_t> + D * u_t

with ``a_t = exp(delta_t * A)``, so all ``a`` live in ``(0, 1]``. The pair
``(a, b)`` composes associatively -- ``(a1, b1) . (a2, b2) = (a1*a2, a2*b1 + b2)``
-- which is what makes the recurrence parallelisable as a prefix scan.

Three tiers, resolved lazily on first use (so a broken or slow optional backend
can never delay or break ComfyUI startup):

===== ============================== ==========================================
tier  backend                        selected when
===== ============================== ==========================================
0     ``mamba_ssm`` CUDA kernel      importable and tensors are on CUDA
1     fused Triton kernel            ``triton`` importable **and** the probe finds
                                     it both correct and faster
2     pure-PyTorch chunked scan      always; the reference-quality fallback
===== ============================== ==========================================

The contract is exactly ``mamba_ssm.ops.selective_scan_interface.selective_scan_fn``
(see :func:`selective_scan_ref` for the definition).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("MambaIR")

#: Tokens per chunk in the pure-PyTorch scan. Tuned on a 6 GB laptop; re-tune with tests/bench.py.
CHUNK = 256

#: device.type -> tier, resolved once per device: a CPU scan must not demote a later
#: CUDA scan (BUG-1). ``_probe_state`` stays machine-global on purpose -- "is this kernel
#: good on this GPU" does not depend on which tensor arrived first.
_resolved: dict[str, str] = {}
_mamba_fn = None
_triton_fn = None
_probe_state = "pending"  # pending -> adopted | rejected
_triton_ok = False
#: what actually ran most recently, for reporting only; ``None`` until the first call
_last_backend: str | None = None
#: benchmarks pin a tier instead of letting the probe decide
_forced: str | None = None
PROBE_TOKENS = 16_384
#: Half the probe length. Below this, the two tiers are milliseconds apart and launch noise can
#: decide -- and a probe verdict is permanent, so a small first call (a 32 px tile is only 1024
#: tokens) must not make it. Such calls run on the reference tier, which is also what makes small
#: images bit-reproducible regardless of GPU: the tiers differ by ~1e-7, so which one runs is
#: observable. Real tiles (128 px = 16 384 tokens) always clear this.
PROBE_MIN_TOKENS = PROBE_TOKENS // 2


# --------------------------------------------------------------------------- #
# backend resolution
# --------------------------------------------------------------------------- #
def _import_mamba_ssm():
    """The ``mamba_ssm`` kernel, or ``None`` when it is unavailable for any reason."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as fn

        return fn
    except Exception:  # noqa: BLE001 - any import/runtime failure means "unavailable"
        return None


def _import_triton():
    """Our fused Triton kernel, or ``None``; logs why when it cannot be imported."""
    try:
        from .scan_triton import selective_scan_triton as fn

        return fn
    except Exception as exc:  # noqa: BLE001 - triton missing, or kernel source rejected
        # Distinguish "absent" from "broken": same behaviour, different diagnosis.
        log.info("MambaIR: Triton scan unavailable (%s); using the PyTorch scan", exc)
        return None


def backend_name() -> str:
    """Human-readable name of the backend that most recently ran (resolves nothing)."""
    if _last_backend is None:
        return "pending"
    return "triton (fused)" if _last_backend == "triton" else _last_backend


def reset_backend_for_tests() -> None:
    """Forget every resolved backend and probe verdict (tests only)."""
    global _mamba_fn, _triton_fn, _probe_state, _triton_ok, _last_backend, _forced
    _resolved.clear()
    _mamba_fn = _triton_fn = None
    _probe_state, _triton_ok = "pending", False
    _last_backend = _forced = None


def force_backend_for_tests(name: str | None, *, trust: bool = False) -> None:
    """Pin a tier regardless of device, or clear the pin with ``None``.

    ``trust`` also marks the tier as adopted, skipping the probe. Benchmark baselines and golden
    captures need that: a recorded output must not depend on how one timing sample came out.
    """
    global _forced, _last_backend, _probe_state, _triton_ok, _triton_fn, _mamba_fn
    _forced = name
    if name is not None:
        _last_backend = name
    if name == "triton" and _triton_fn is None:
        _triton_fn = _import_triton()   # forcing a tier must also load it
    if name == "mamba_ssm" and _mamba_fn is None:
        _mamba_fn = _import_mamba_ssm()
    if trust and name in ("triton", "mamba_ssm"):
        _probe_state, _triton_ok = "adopted", True


def _demote_triton(reason) -> None:
    """Retire the Triton tier in one place, so no state can describe it as active (BUG-2)."""
    global _triton_ok, _probe_state, _triton_fn
    _triton_ok, _probe_state, _triton_fn = False, "rejected", None
    if _resolved.get("cuda") != "mamba_ssm":
        _resolved["cuda"] = "torch"
    log.warning("MambaIR: Triton scan disabled (%s); using the PyTorch scan", reason)


def _resolve(u: torch.Tensor) -> str:
    """Pick the fastest available tier for this tensor's device, caching per device type.

    ``mamba_ssm`` wins when importable; otherwise the fused Triton kernel is loaded and the probe
    decides later. CPU tensors always resolve to the PyTorch tier.
    """
    global _mamba_fn, _triton_fn, _last_backend
    if _forced is not None:
        _last_backend = _forced
        return _forced
    kind = u.device.type
    backend = _resolved.get(kind)
    if backend is None:
        backend = "torch"
        if kind == "cuda":
            _mamba_fn = _import_mamba_ssm()
            if _mamba_fn is not None:
                backend = "mamba_ssm"
            else:
                _triton_fn = _import_triton()
        _resolved[kind] = backend
    _last_backend = backend
    return backend


# --------------------------------------------------------------------------- #
# shared preparation (all tiers agree on this)
# --------------------------------------------------------------------------- #
def _prep(u, delta, A, B, C, delta_bias, delta_softplus):
    """Broadcast to ``(b, d, l, n)`` and discretise. Everything is float32."""
    b, d, l = u.shape
    dtype = torch.float32
    u = u.to(dtype)
    delta = delta.to(dtype)
    A = A.to(dtype)
    if delta_bias is not None:
        delta = delta + delta_bias.to(dtype).view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    def bc(t):
        t = t.to(dtype)
        if t.dim() == 3:  # (b, n, l) -> one group for all channels
            return t.unsqueeze(1).expand(b, d, t.shape[1], t.shape[2])
        g = t.shape[1]  # (b, g, n, l)
        if g == 1:
            return t.expand(b, d, t.shape[2], t.shape[3])
        if d % g:
            raise ValueError(f"groups ({g}) must divide channels ({d})")
        return t.unsqueeze(2).expand(b, g, d // g, t.shape[2], t.shape[3]).reshape(
            b, d, t.shape[2], t.shape[3]
        )

    B = bc(B).transpose(-1, -2)  # (b, d, l, n)
    C = bc(C).transpose(-1, -2)
    # A is (d, n): broadcast over batch and sequence, never over channels
    dA = torch.exp(delta.unsqueeze(-1) * A.view(1, A.shape[0], 1, A.shape[1]))
    dBu = (delta.unsqueeze(-1) * B) * u.unsqueeze(-1)
    return u, dA, dBu, C


def _finish(y, u, D, z):
    """Apply the skip connection and the optional SiLU gate to a raw scan output."""
    if D is not None:
        y = y + u * D.to(y.dtype).view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z.to(y.dtype))
    return y


# --------------------------------------------------------------------------- #
# tier 2: pure PyTorch
# --------------------------------------------------------------------------- #
def _doubling(a, q, carry):
    """Hillis-Steele doubling for one chunk. ``a``/``q`` are modified in place.

    ``a``: ``(b, d, Lc, n)`` decay factors, ``q``: ``(b, d, Lc, n)`` inputs,
    ``carry``: ``(b, d, n)`` state entering the chunk. Returns the state after
    the chunk.
    """
    L = a.shape[2]
    q[:, :, 0] = q[:, :, 0] + a[:, :, 0] * carry
    step = 1
    while step < L:
        a_hi = a[:, :, step:]
        a_lo = a[:, :, : L - step]
        b_hi = q[:, :, step:]
        b_lo = q[:, :, : L - step]
        # a_hi/a_lo overlap: materialise before writing back, or lengths that are not
        # powers of two are corrupted (see tests/test_scan.py::test_scan_torch_matches_reference).
        b_hi += a_hi * b_lo
        a[:, :, step:] = a_hi * a_lo
        step *= 2
    return q[:, :, -1]


def _scan_torch(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, chunk: int = CHUNK):
    """Chunked associative scan with the discretisation computed *inside* the loop.

    This is the memory-critical detail: ``exp(delta*A)``, ``delta*B*u`` and ``C``
    are all ``(b, d, L, n)`` tensors, so building them for a whole sequence costs
    gigabytes at Large width. Computed per chunk the peak stays flat, and the
    output is bit-identical (see tests/test_scan.py::test_chunk_local_prep_is_bit_identical).

    Note that ``B`` and ``C`` are usually *shared* across channels in these models
    (``(b, 1, n, L)`` for all channels), so broadcasting them instead of materialising them per
    channel removes a large inflation.

    Args:
        u: Input, shape ``(batch, channels, length)``.
        delta: Step sizes, shape ``(batch, channels, length)``.
        A: State transition, shape ``(channels, d_state)``.
        B: Input projection, ``(batch, groups, d_state, length)`` or ``(batch, d_state, length)``.
        C: Output projection, same layout as ``B``.
        D: Per-channel skip weight, shape ``(channels,)``, or ``None``.
        z: Gate for the SiLU-multiplied variant, shape ``(batch, channels, length)``, or ``None``.
        delta_bias: Per-channel bias added to ``delta`` before ``softplus``, shape ``(channels,)``.
        delta_softplus: Apply ``softplus`` to ``delta``.
        return_last_state: Also return the final state.
        chunk: Tokens per chunk; the discretisation is built and scanned chunk by chunk.

    Returns:
        Output of shape ``(batch, channels, length)``, or ``(output, last_state)``.

    Raises:
        ValueError: If ``groups`` does not divide ``channels``.
    """
    b, d, l = u.shape
    n = A.shape[1]
    u = u.to(torch.float32)
    delta = delta.to(torch.float32)
    A = A.to(torch.float32)
    Bc = B if B.dim() == 4 else B.unsqueeze(1)  # (b, g, n, l)
    Cc = C if C.dim() == 4 else C.unsqueeze(1)
    groups = Bc.shape[1]
    if d % groups:
        raise ValueError(f"groups ({groups}) must divide channels ({d})")

    y = torch.empty(b, d, l, dtype=torch.float32, device=u.device)
    Acol = A.view(1, d, 1, n)
    carry = torch.zeros(b, d, n, dtype=torch.float32, device=u.device)

    for start in range(0, l, chunk):
        stop = min(start + chunk, l)
        dt = delta[:, :, start:stop]
        if delta_bias is not None:
            dt = dt + delta_bias.to(torch.float32).view(1, -1, 1)
        if delta_softplus:
            dt = F.softplus(dt)
        uc = u[:, :, start:stop]

        # Broadcast over channels instead of expanding, unless there are real groups.
        Bt = Bc[:, :, :, start:stop].transpose(-1, -2)  # (b, g, Lc, n)
        Ct = Cc[:, :, :, start:stop].transpose(-1, -2)
        if groups != d:
            rep = d // groups
            Bt = Bt.unsqueeze(2).expand(b, groups, rep, Bt.shape[2], n).reshape(b, d, Bt.shape[2], n)
            Ct = Ct.unsqueeze(2).expand(b, groups, rep, Ct.shape[2], n).reshape(b, d, Ct.shape[2], n)

        a = torch.exp(dt.unsqueeze(-1) * Acol)
        q = (dt.unsqueeze(-1) * Bt) * uc.unsqueeze(-1)
        carry = _doubling(a, q, carry)
        y[:, :, start:stop] = (q * Ct).sum(-1)

    y = _finish(y, u, D, z)
    return (y, carry) if return_last_state else y


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Run the scan one token at a time, as the operator is defined.

    One Python step per token, with every intermediate materialised: ``(b, d, l, n)`` tensors for
    ``exp(delta*A)``, ``delta*B*u`` and ``C``. That is why the runtime never uses it -- it exists as
    the oracle the other tiers are checked against, and it takes the same arguments as
    :func:`selective_scan_fn`.

    Args:
        u: Input, shape ``(batch, channels, length)``.
        delta: Step sizes, shape ``(batch, channels, length)``.
        A: State transition, shape ``(channels, d_state)``.
        B: Input projection, ``(batch, groups, d_state, length)`` or ``(batch, d_state, length)``.
        C: Output projection, same layout as ``B``.
        D: Per-channel skip weight, shape ``(channels,)``, or ``None``.
        z: Gate for the SiLU-multiplied variant, shape ``(batch, channels, length)``, or ``None``.
        delta_bias: Per-channel bias added to ``delta`` before ``softplus``, shape ``(channels,)``.
        delta_softplus: Apply ``softplus`` to ``delta``.
        return_last_state: Also return the final state.

    Returns:
        Output of shape ``(batch, channels, length)``, or ``(output, last_state)``.

    Raises:
        ValueError: If ``groups`` does not divide ``channels``.
    """
    u, dA, dBu, Cp = _prep(u, delta, A, B, C, delta_bias, delta_softplus)
    b, d, l, n = dA.shape
    x = torch.zeros(b, d, n, dtype=torch.float32, device=u.device)
    ys = []
    for t in range(l):
        x = dA[:, :, t] * x + dBu[:, :, t]
        ys.append((x * Cp[:, :, t]).sum(-1))
    y = _finish(torch.stack(ys, dim=-1), u, D, z)
    return (y, x) if return_last_state else y


# --------------------------------------------------------------------------- #
# tier 1: Triton, adopted only if the probe says it is correct *and* faster
# --------------------------------------------------------------------------- #
def _probe_tensors(u, delta, A, B, C, D, z, delta_bias):
    """Deterministic tensors with the shapes the user's call would use.

    The probe's verdict is permanent, so it must not depend on image content: model
    activations from a plain black image are enough to separate the two tiers by more
    than the correctness cut, which rejected a kernel that was *closer* to the
    double-precision oracle than the reference tier (see design.md §11b, D-5). The
    distribution matches ``tests/test_scan_triton.py``'s fixed random inputs.
    """
    g = torch.Generator(device=u.device).manual_seed(0)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=u.device, dtype=torch.float32) * scale

    L = PROBE_TOKENS
    shapes = {}
    for name, tensor in (("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C),
                         ("z", z), ("D", D), ("bias", delta_bias)):
        shapes[name] = None if tensor is None else tuple(tensor.shape)

    pu = rand(shapes["u"][0], shapes["u"][1], L, scale=3.0)
    pdelta = torch.rand(shapes["delta"][0], shapes["delta"][1], L,
                        generator=g, device=u.device, dtype=torch.float32) * 0.5
    pA = -torch.exp(rand(*shapes["A"]))
    pB = rand(*shapes["B"][:-1], L)
    pC = rand(*shapes["C"][:-1], L)
    pz = rand(*shapes["z"][:-1], L) if z is not None else None
    pD = rand(*shapes["D"]) if D is not None else None
    pbias = rand(*shapes["bias"]) if delta_bias is not None else None
    return pu, pdelta, pA, pB, pC, pD, pz, pbias


def _probe_triton(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    """Check correctness, then time; adopt or reject the tier permanently.

    Both checks run on deterministically generated tensors with the shapes of the user's
    first call, at fixed length :data:`PROBE_TOKENS`. The decision therefore depends on
    the machine's kernel, not on the image the user happened to load first.
    """
    global _probe_state, _triton_ok
    if _triton_fn is None or _probe_state != "pending":
        return _triton_ok
    _probe_state = "rejected"
    try:
        u, delta, A, B, C, D, z, delta_bias = _probe_tensors(
            u, delta, A, B, C, D, z, delta_bias
        )

        import time

        # Triton JIT-compiles the kernel on first launch, which would dominate the
        # measurement, so warm it up on an 8-token slice first. Otherwise the timing
        # below compares a compile against a compiled reference and rejects a good kernel.
        head = slice(0, 8)
        _triton_fn(
            u[..., head], delta[..., head], A, B[..., head], C[..., head], D,
            z[..., head] if z is not None else None, delta_bias, delta_softplus,
        )
        torch.cuda.synchronize()

        # Time the correctness passes rather than running each tier a second time: the probe
        # is a one-time cost on the first forward, so it must not repeat work.
        t0 = time.perf_counter()
        ref = _scan_torch(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        torch.cuda.synchronize()
        t_torch = time.perf_counter() - t0

        t0 = time.perf_counter()
        got = _triton_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        torch.cuda.synchronize()
        t_triton = time.perf_counter() - t0

        # Scale-aware: a 16k-step sequential accumulation drifts by ~1e-4 in
        # absolute terms even when it is 1e-7 relative, so `atol` alone rejects a
        # correct kernel. Compare against the magnitude of the result.
        ref_f, got_f = ref.float(), got.float()
        if not torch.isfinite(got_f).all():
            log.warning("MambaIR: Triton scan produced non-finite values; using the PyTorch scan")
            return False
        error = (got_f - ref_f).abs().max().item()
        scale = max(ref_f.abs().max().item(), 1e-3)
        if error / scale > 1e-4:
            log.warning(
                "MambaIR: Triton scan disagrees with the reference (relative error %.2e); using the PyTorch scan",
                error / scale,
            )
            return False
        if t_triton * 1.05 >= t_torch:
            # One retry, only on a loss. The verdict is permanent, and a single sample can carry
            # a first-touch allocation stall of the same order as the kernel itself (measured: a
            # 0.50 s outlier for a call that otherwise takes 1-14 ms), which flips the decision
            # from run to run and would leave the process on the reference tier for good. The
            # adopted path still times each tier exactly once (BUG-18).
            t0 = time.perf_counter()
            _triton_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            torch.cuda.synchronize()
            t_triton = min(t_triton, time.perf_counter() - t0)
        if t_triton * 1.05 >= t_torch:
            log.info(
                "MambaIR: Triton scan is not faster (%.3f s vs %.3f s at %d tokens); keeping the PyTorch scan",
                t_triton, t_torch, PROBE_TOKENS,
            )
            return False
        _probe_state, _triton_ok = "adopted", True
        log.info(
            "MambaIR: Triton scan adopted (%.3f s vs %.3f s at %d tokens) for all sequence lengths",
            t_triton, t_torch, PROBE_TOKENS,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - any kernel failure means "fall back"
        log.warning("MambaIR: Triton scan failed (%s); using the PyTorch scan", exc)
        return False


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Run a selective scan.

    Resolves a backend on first use: ``mamba_ssm`` if importable, else the fused Triton kernel if
    the probe accepts it, else the chunked PyTorch scan. Calls below ``PROBE_MIN_TOKENS`` stay on
    the PyTorch tier, so a small first call cannot decide the tier for the process.

    Args:
        u: Input, shape ``(batch, channels, length)``.
        delta: Step sizes, shape ``(batch, channels, length)``.
        A: State transition, shape ``(channels, d_state)``; negative by convention.
        B: Input projection, ``(batch, groups, d_state, length)``, or ``(batch, d_state, length)``
            for a single shared group.
        C: Output projection, same layout as ``B``.
        D: Per-channel skip weight, shape ``(channels,)``, or ``None``.
        z: Gate for the SiLU-multiplied variant, shape ``(batch, channels, length)``, or ``None``.
        delta_bias: Per-channel bias added to ``delta`` before ``softplus``, shape ``(channels,)``.
        delta_softplus: Apply ``softplus`` to ``delta``.
        return_last_state: Also return the final state.

    Returns:
        Output of shape ``(batch, channels, length)``, or ``(output, last_state)`` with
        ``last_state`` of shape ``(batch, channels, d_state)`` when ``return_last_state`` is set.

    Raises:
        ValueError: If ``groups`` does not divide ``channels``.
    """
    global _last_backend
    backend = _resolve(u)

    if backend == "mamba_ssm" and _mamba_fn is not None:
        try:
            out = _mamba_fn(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
                delta_softplus=delta_softplus, return_last_state=return_last_state,
            )
            _last_backend = "mamba_ssm"
            return out
        except Exception as exc:  # noqa: BLE001 - e.g. unsupported dtype/device
            _resolved["cuda"] = "torch"
            log.warning("MambaIR: mamba_ssm kernel failed (%s); using the PyTorch scan", exc)

    if (
        _triton_fn is not None
        and u.is_cuda
        and _forced in (None, "triton")
        and u.shape[-1] >= PROBE_MIN_TOKENS
        and _probe_triton(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    ):
        try:
            out = _triton_fn(
                u, delta, A, B, C, D, z, delta_bias, delta_softplus,
                return_last_state=return_last_state,
            )
            _last_backend = "triton"
            return out
        except Exception as exc:  # noqa: BLE001 - permanent demotion
            _demote_triton(exc)

    _last_backend = "torch"
    return _scan_torch(
        u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
        delta_softplus=delta_softplus, return_last_state=return_last_state,
    )
