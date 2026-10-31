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


"""Regression tests for the hardening pass.

One named test per defect (``BUG-n``), so the fix and its proof share an identifier. Tests are
grouped by the file they defend and are added wave by wave.
"""

import contextlib
import logging
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import corpus
from mambair import node_core, scan, tiling
from mambair.apply import apply_descriptor


@contextlib.contextmanager
def captured(level=logging.INFO):
    """Collect log records from the MambaIR logger."""
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("MambaIR")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _inputs(b=1, d=4, length=32, n=4, device="cpu", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    u = torch.randn(b, d, length, generator=gen, device=device)
    delta = torch.rand(b, d, length, generator=gen, device=device) * 0.5
    A = -torch.exp(torch.randn(d, n, generator=gen, device=device))
    B = torch.randn(b, 1, n, length, generator=gen, device=device)
    C = torch.randn(b, 1, n, length, generator=gen, device=device)
    return u, delta, A, B, C


# --------------------------------------------------------------------------- #
# BUG-1 / EXP-1: backend resolution is per device, not per process
# --------------------------------------------------------------------------- #
def test_bug_1_cpu_scan_does_not_demote_a_later_cuda_scan():
    if not torch.cuda.is_available():
        return "skip"
    scan.reset_backend_for_tests()
    try:
        with torch.inference_mode():
            scan.selective_scan_fn(*_inputs(), delta_softplus=True)
        assert scan._resolved.get("cpu") == "torch", scan._resolved

        with torch.inference_mode():
            scan.selective_scan_fn(*_inputs(d=32, length=scan.PROBE_TOKENS, device="cuda"),
                                   delta_softplus=True)

        best = "mamba_ssm" if scan._mamba_fn is not None else "torch"
        assert scan._resolved.get("cuda") == best, scan._resolved
        if scan._mamba_fn is None and scan._triton_fn is not None:
            assert scan.backend_name() == "triton (fused)", scan.backend_name()
    finally:
        scan.reset_backend_for_tests()


def test_bug_1_cuda_then_cpu_keeps_both_resolutions():
    """P-DEV-1: either order yields the best tier for the device that is asking."""
    if not torch.cuda.is_available():
        return "skip"
    scan.reset_backend_for_tests()
    try:
        with torch.inference_mode():
            scan.selective_scan_fn(*_inputs(d=32, length=scan.PROBE_TOKENS, device="cuda"),
                                   delta_softplus=True)
        cuda_tier = scan._resolved.get("cuda")
        with torch.inference_mode():
            scan.selective_scan_fn(*_inputs(), delta_softplus=True)
        assert scan._resolved.get("cpu") == "torch"
        assert scan._resolved.get("cuda") == cuda_tier, "a CPU scan must not change the CUDA tier"
        assert scan._triton_fn is not None or scan._mamba_fn is not None
    finally:
        scan.reset_backend_for_tests()


# --------------------------------------------------------------------------- #
# BUG-2 / EXP-2: one demotion path; reporting follows what actually ran
# --------------------------------------------------------------------------- #
def test_bug_2_triton_failure_demotes_all_state():
    if not torch.cuda.is_available():
        return "skip"
    scan.reset_backend_for_tests()
    try:
        def boom(*args, **kwargs):
            raise RuntimeError("injected kernel failure")

        # pretend the probe already adopted, and that resolution ran
        scan._resolved["cuda"] = "torch"
        scan._probe_state, scan._triton_ok, scan._triton_fn = "adopted", True, boom

        # a representative length: below PROBE_MIN_TOKENS the Triton tier is not consulted
        args = _inputs(d=32, length=scan.PROBE_TOKENS, device="cuda")
        with torch.inference_mode():
            got = scan.selective_scan_fn(*args, delta_softplus=True)
            ref = scan._scan_torch(*args, delta_softplus=True)

        assert scan.backend_name() == "torch", scan.backend_name()
        assert scan._triton_ok is False
        assert scan._triton_fn is None, "a retired kernel must not be retried"
        assert scan._probe_state == "rejected"
        assert torch.allclose(got, ref, atol=1e-5)
    finally:
        scan.reset_backend_for_tests()


# --------------------------------------------------------------------------- #
# BUG-9 / EXP-9: an unimportable Triton must be diagnosable
# --------------------------------------------------------------------------- #
def test_bug_9_import_failure_is_logged():
    saved = sys.modules.pop("mambair.scan_triton", None)
    sys.modules["mambair.scan_triton"] = None  # forces ImportError for the relative import
    try:
        with captured(logging.INFO) as records:
            fn = scan._import_triton()
        assert fn is None
        messages = [r.getMessage() for r in records]
        assert any("Triton scan unavailable" in m for m in messages), messages
    finally:
        del sys.modules["mambair.scan_triton"]
        if saved is not None:
            sys.modules["mambair.scan_triton"] = saved


# --------------------------------------------------------------------------- #
# BUG-14 / EXP-14: no unreachable backend-selection conditions
# --------------------------------------------------------------------------- #
def test_bug_14_size_threshold_knob_is_gone():
    assert not hasattr(scan, "_TRITON_MAX_TOKENS"), "the removed threshold must not linger"
    assert isinstance(scan._triton_ok, bool)
    scan.reset_backend_for_tests()
    assert scan._triton_ok is False
    assert scan._resolved == {}
    assert scan.backend_name() == "pending"


# --------------------------------------------------------------------------- #
# BUG-3 / EXP-3: every path runs without autograd
# --------------------------------------------------------------------------- #
class Upscaler(nn.Module):
    """Stand-in for a foreign upscale model: nearest upsample, then a conv with weights."""

    def __init__(self, true_scale=2, declared=None):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)
        self.true_scale = true_scale
        self.scale = true_scale if declared is None else declared

    def forward(self, x):
        if self.true_scale != 1:
            x = F.interpolate(x, scale_factor=self.true_scale, mode="nearest")
        return self.conv(x).clamp(0, 1)


class RunnerDescriptor:
    """Minimal stand-in for a spandrel descriptor carrying a MambaIR runner."""

    def __init__(self, runner):
        self._mambair_runner = runner
        self.model = runner.arch
        self.scale = runner.scale


def test_bug_3_generic_path_is_grad_free():
    model = Upscaler(true_scale=2)
    seen = {}

    def record_grad_state(_module, _inputs_):  # pre-hooks must return None to leave inputs alone
        seen["grad"] = torch.is_grad_enabled()

    model.conv.register_forward_pre_hook(record_grad_state)
    out, _status = apply_descriptor(model, torch.rand(1, 32, 32, 3))
    assert out.requires_grad is False
    assert seen["grad"] is False, "the forward must not run with grad enabled"


def test_bug_3_mambair_path_is_grad_free_too():
    out, _status = apply_descriptor(RunnerDescriptor(corpus.tiny_runner()), torch.rand(1, 16, 16, 3))
    assert out.requires_grad is False


# --------------------------------------------------------------------------- #
# BUG-4 / EXP-4: the generic branch uses the same OOM ladder
# --------------------------------------------------------------------------- #
class OOMUpscaler(Upscaler):
    def __init__(self, fail_times, true_scale=1):
        super().__init__(true_scale=true_scale)
        self.fail_times = fail_times
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise torch.cuda.OutOfMemoryError("injected out-of-memory")
        return super().forward(x)


def test_bug_4_generic_path_halves_on_oom():
    model = OOMUpscaler(fail_times=1)
    image = torch.rand(1, 600, 600, 3)
    with captured(logging.WARNING) as records:
        out, _status = apply_descriptor(model, image, tile=512, overlap=32)
    assert out.shape == (1, 600, 600, 3), out.shape
    assert model.calls > 4, "the retry must re-run the whole image, not just one tile"
    assert any("out of memory" in r.getMessage() for r in records), [r.getMessage() for r in records]


def test_bug_4_ladder_stops_at_the_floor():
    model = OOMUpscaler(fail_times=999)
    image = torch.rand(1, 600, 600, 3)
    try:
        apply_descriptor(model, image, tile=512, overlap=32)
    except torch.cuda.OutOfMemoryError:
        pass
    else:
        raise AssertionError("an always-OOM model must raise after the ladder bottoms out")
    assert model.calls == 3, f"expected attempts at 512/256/128, got {model.calls}"


# --------------------------------------------------------------------------- #
# BUG-5 / EXP-5: the blend derives the true scale instead of trusting the label
# --------------------------------------------------------------------------- #
def test_bug_5_scale_mismatch_recovers():
    """Covers both the single-tile fast path and the blending path."""
    for actual, declared in [(1, 2), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2)]:
        model = Upscaler(true_scale=actual, declared=declared)
        for size, tile in ((40, 0), (400, 64)):  # one tile, then many
            with captured(logging.WARNING) as records:
                out, status = apply_descriptor(model, torch.rand(1, size, size, 3), tile=tile)
            assert out.shape == (1, size * actual, size * actual, 3), (actual, declared, size, out.shape)
            assert f"x{actual}" in status, (status, size)
            assert any("reports x" in r.getMessage() for r in records), [r.getMessage() for r in records]


# --------------------------------------------------------------------------- #
# BUG-7 / EXP-7: the cap follows the device, and the progress total follows the cap
# --------------------------------------------------------------------------- #
def test_bug_7_cap_matches_device():
    assert tiling.default_tile_cap(torch.device("cpu")) == 256
    assert tiling.default_tile_cap(torch.device("cuda")) == 320
    assert tiling.default_tile_cap() == (320 if torch.cuda.is_available() else 256)

    image = torch.zeros(1, 700, 900, 3)
    assert node_core._estimate_steps(image, 0, 32) == tiling.tile_count(
        700, 900, tiling.default_tile_cap(), 32
    )


def test_bug_7_progress_total_matches_the_real_grid():
    # sizes chosen because the 256 px and 320 px grids differ there (25 vs 16, 18 vs 15):
    # a size where both caps happen to give the same count would pass without testing anything.
    for height, width in ((1000, 1000), (1300, 700)):
        image = torch.zeros(1, height, width, 3)
        calls = {"n": 0}

        def fn(t):
            calls["n"] += 1
            return t

        tiling.tiled_forward(fn, image.movedim(-1, -3), 1, tiling.default_tile_cap(), 32)
        total = node_core._estimate_steps(image, 0, 32)
        assert total == calls["n"], f"{height}x{width}: progress total {total} != {calls['n']} tiles"


# --------------------------------------------------------------------------- #
# BUG-8 / EXP-8: overlap >= tile cannot explode the tile count
# --------------------------------------------------------------------------- #
def test_bug_8_overlap_ge_tile_is_bounded():
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        return t

    x = torch.rand(1, 3, 512, 512)
    with captured(logging.DEBUG):
        out = tiling.tiled_forward(fn, x, 1, 128, 200)
    assert out.shape == x.shape
    bound = tiling.tile_count(512, 512, 128, 64)
    assert calls["n"] <= bound, f"{calls['n']} tiles at overlap>=tile; the clamp allows {bound}"


# --------------------------------------------------------------------------- #
# BUG-11 / EXP-11: 1- and 2-channel images do not reach a 3-channel conv
# --------------------------------------------------------------------------- #
def test_bug_11_channel_expansion():
    for channels in (1, 2, 3):
        out, _status = apply_descriptor(Upscaler(true_scale=2), torch.rand(1, 32, 32, channels))
        assert out.shape == (1, 64, 64, 3), (channels, out.shape)


def test_bug_11_alpha_is_preserved_at_size():
    image = torch.rand(1, 32, 32, 4)
    image[..., 3] = 0.5
    out, _status = apply_descriptor(Upscaler(true_scale=2), image)
    assert out.shape == (1, 64, 64, 4), out.shape
    assert torch.allclose(out[..., 3], torch.full((1, 64, 64), 0.5)), "alpha must be resampled"


def test_bug_11_grayscale_input_on_a_one_channel_checkpoint():
    """A 1-channel IMAGE is expanded; the 1-channel *checkpoint* still uses the luma policy."""
    out, _status = apply_descriptor(RunnerDescriptor(corpus.tiny_runner()), torch.rand(1, 16, 16, 1))
    assert out.shape == (1, 32, 32, 3), out.shape


# --------------------------------------------------------------------------- #
# BUG-16 / EXP-16: one full-size pass after the blend
# --------------------------------------------------------------------------- #
def test_bug_16_single_full_size_pass():
    if not torch.cuda.is_available():
        return "skip"
    x = torch.rand(1, 3, 512, 512, device="cuda")

    def fn(t):
        return F.interpolate(t, scale_factor=1, mode="nearest")

    tiling.tiled_forward(fn, x, 1, 256, 32)  # warm up the allocator
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = tiling.tiled_forward(fn, x, 1, 256, 32)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline

    ratio = peak / (out.numel() * out.element_size())
    # output buffer + accumulator + per-tile temporaries; a separate division result
    # would add another full-size buffer on top.
    assert ratio < 2.3, f"peak is {ratio:.2f}x the output; the division is not in place"


# --------------------------------------------------------------------------- #
# BUG-17 was withdrawn after measurement: a cached mask is *slower* than rebuilding
# it per tile (5.2/5.9/5.6 ms vs 3.2/4.3/4.2 ms for 16 masks at scale 1/2/4), because
# the work is one 0.26 MB host->device copy against three device launches. The
# revert is the fix, so the guard lives in test_tiling.py's seam tests.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# BUG-6 / EXP-6: an unimplemented upsampler must fail loudly, not silently
# --------------------------------------------------------------------------- #
def _v1_state_dict_with_realworld_upsampler():
    return {
        "conv_first.weight": torch.zeros(32, 3, 3, 3),
        "conv_up1.0.weight": torch.zeros(32, 32, 3, 3),
        "layers.0.residual_group.blocks.0.self_attention.x_proj_weight": torch.zeros(4, 36, 32),
    }


def test_bug_6_v1_realworld_upsampler_is_rejected():
    from mambair import infer as inf

    with captured(logging.DEBUG):
        try:
            inf.infer(_v1_state_dict_with_realworld_upsampler())
        except inf.UnsupportedUpsampler as exc:
            assert "nearest+conv" in str(exc), exc
        else:
            raise AssertionError("v1 has no nearest+conv branch; infer must not claim it does")


def test_bug_6_v2_realworld_upsampler_is_accepted():
    """The v2 families *do* implement it, so the rejection must be per family."""
    from mambair import infer as inf

    sd = {
        "conv_first.weight": torch.zeros(32, 3, 3, 3),
        "conv_up1.0.weight": torch.zeros(32, 32, 3, 3),
        "conv_up2.0.weight": torch.zeros(32, 32, 3, 3),
        "layers.0.residual_group.layers.0.assm.selectiveScan.A_logs": torch.zeros(64, 4),
        "layers.0.residual_group.layers.0.assm.embeddingB.weight": torch.zeros(8, 8),
        "layers.0.residual_group.layers.0.assm.in_proj.0.weight": torch.zeros(64, 32, 1, 1),
        "layers.0.residual_group.layers.0.convffn1.sg.conv.weight": torch.zeros(32, 1, 5, 5),
        "layers.0.residual_group.layers.0.win_mhsa.relative_position_bias_table": torch.zeros(49, 2),
    }
    hp = inf.infer(sd)
    assert (hp.upscale, hp.upsampler) == (4, "nearest+conv"), (hp.upscale, hp.upsampler)


def test_bug_6_spandrel_reports_an_unsupported_model():
    from spandrel import UnsupportedModelError

    from mambair import registry
    from mambair.spandrel_arch import _load

    registry.register_all()
    try:
        _load(_v1_state_dict_with_realworld_upsampler(), None, "MambaIR")
    except UnsupportedModelError as exc:
        assert "nearest+conv" in str(exc) and "MambaIR" in str(exc), exc
    else:
        raise AssertionError("loading must fail with UnsupportedModelError")


# --------------------------------------------------------------------------- #
# BUG-10 / EXP-10: seeds and cache keys are unique
# --------------------------------------------------------------------------- #
def test_bug_10_seed_uniqueness():
    prints = {node_core.fingerprint(seed=-1) for _ in range(200)}
    assert len(prints) == 200, f"{200 - len(prints)} collisions in 200 cache keys"
    assert all(isinstance(p, int) for p in prints), "nanosecond keys must not be floats"


def test_bug_10_random_seed_changes_the_result():
    image = torch.rand(1, 16, 16, 3)
    first, _ = node_core.run_node(RunnerDescriptor(tiny := corpus.tiny_runner()), image, seed=-1)
    second, _ = node_core.run_node(RunnerDescriptor(tiny), image, seed=-1)
    assert not torch.equal(first, second), "seed=-1 must not reuse a previous run's routing"


# --------------------------------------------------------------------------- #
# BUG-12 / EXP-12: a tile below the floor is reported, not silently raised
# --------------------------------------------------------------------------- #
def test_bug_12_tile_below_floor_is_reported():
    image = torch.zeros(1, 400, 400, 3)
    # the effective cap is the floor, and the progress total says so
    assert node_core._estimate_steps(image, 64, 32) == tiling.tile_count(400, 400, tiling.MIN_CAP, 32)
    with captured(logging.WARNING) as records:
        node_core.run_node(RunnerDescriptor(corpus.tiny_runner()), image, tile=64, overlap=32)
    messages = [r.getMessage() for r in records]
    assert any("below the" in m for m in messages), messages


# --------------------------------------------------------------------------- #
# BUG-13 / EXP-13: the tooltip cannot drift from the code
# --------------------------------------------------------------------------- #
def test_bug_13_tooltip_matches_cap():
    numbers = {int(n) for n in re.findall(r"\d+", node_core.TILE_TOOLTIP)}
    assert tiling.default_tile_cap(torch.device("cuda")) in numbers, node_core.TILE_TOOLTIP
    assert tiling.default_tile_cap(torch.device("cpu")) in numbers, node_core.TILE_TOOLTIP
    assert tiling.MIN_CAP in numbers, node_core.TILE_TOOLTIP


# --------------------------------------------------------------------------- #
# BUG-15 / EXP-15: dead code and escaping mutable state
# --------------------------------------------------------------------------- #
def test_bug_15_no_orphaned_helpers():
    from mambair import infer as inf
    from mambair import registry

    assert not hasattr(inf, "_has"), "orphaned helper still present"
    assert not hasattr(inf, "_keys"), "orphaned helper still present"
    first = registry.register_all()
    assert isinstance(first, tuple), type(first)
    assert registry.register_all() == first


# --------------------------------------------------------------------------- #
# BUG-18 / EXP-18: the probe does not run each tier twice
# --------------------------------------------------------------------------- #
_REAL_IMPORT_TRITON = scan._import_triton


def test_bug_18_probe_runs_each_tier_once():
    if not torch.cuda.is_available():
        return "skip"
    real_triton = _REAL_IMPORT_TRITON()
    if real_triton is None:
        return "skip"
    real_torch = scan._scan_torch
    calls = {"torch": 0, "triton": []}

    def counting_triton(u, delta, A, B, C, D=None, z=None, db=None, ds=False, **kw):
        calls["triton"].append(u.shape[-1])
        return real_triton(u, delta, A, B, C, D, z, db, ds, **kw)

    def counting_torch(*args, **kwargs):
        calls["torch"] += 1
        return real_torch(*args, **kwargs)

    scan._scan_torch = counting_torch
    scan._import_triton = lambda: counting_triton
    adopted = False
    try:
        scan.reset_backend_for_tests()
        with torch.inference_mode():
            scan.selective_scan_fn(*_inputs(d=32, length=scan.PROBE_TOKENS, device="cuda"),
                                   delta_softplus=True)
        adopted = scan.backend_name() == "triton (fused)"   # read before the reset below
    finally:
        scan._scan_torch = real_torch
        scan._import_triton = _REAL_IMPORT_TRITON
        scan.reset_backend_for_tests()
    # a rejection legitimately falls back to the reference tier, which is a second call
    assert calls["torch"] == (1 if adopted else 2), (
        f"reference ran {calls['torch']} times (adopted={adopted}); once is enough"
    )
    # warm-up (8 tokens, triggers JIT compile) -> timed probe call -> the actual scan
    assert calls["triton"][0] <= 8, f"first Triton call should be the compile warm-up: {calls['triton']}"
    if adopted:
        assert calls["triton"] == [8, scan.PROBE_TOKENS, scan.PROBE_TOKENS], calls["triton"]
    else:
        # a loss may be retried once before the verdict is made permanent (see throughput-pass)
        assert calls["triton"][:2] == [8, scan.PROBE_TOKENS], calls["triton"]
        assert len(calls["triton"]) in (2, 3), calls["triton"]


# --------------------------------------------------------------------------- #
# Throughput pass: a permanent tier verdict is not made from one noisy sample
# --------------------------------------------------------------------------- #
def test_probe_retries_a_single_slow_sample():
    if not torch.cuda.is_available():
        return "skip"
    real_triton = _REAL_IMPORT_TRITON()
    if real_triton is None:
        return "skip"
    import time as _time

    calls = []

    def slow_once(u, delta, A, B, C, D=None, z=None, db=None, ds=False, **kw):
        calls.append(u.shape[-1])
        if len(calls) == 2:  # the first timed probe sample, after the 8-token warm-up
            torch.cuda.synchronize()
            _time.sleep(0.5)
        return real_triton(u, delta, A, B, C, D, z, db, ds, **kw)

    try:
        scan.reset_backend_for_tests()
        scan._import_triton = lambda: slow_once  # resolution imports it, so patch the importer
        u, delta, A, B, C = _inputs(1, 8, scan.PROBE_TOKENS, 16, device="cuda")
        with torch.inference_mode():
            scan.selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
        assert scan._probe_state == "adopted", (
            f"a 0.5 s stall in the first sample must not decide the tier: {calls}"
        )
        assert len(calls) == 4, f"warm-up, probe, retry, scan: {calls}"
    finally:
        scan._import_triton = _REAL_IMPORT_TRITON
        scan.reset_backend_for_tests()
