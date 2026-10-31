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

"""`OPT-5` regression tests: per-tile fixed costs.

The model-manager placement is cached per (patcher, device), the failure path still re-loads, and
the output is clamped once and crosses to ComfyUI's device once. The tests inject a fake
``comfy.model_management`` so they run without ComfyUI, and they count calls the same way
``test_scan_triton`` counts kernel launches.
"""

import contextlib
import sys
import types

import torch
import torch.nn.functional as F

from mambair import tiling
from mambair.apply import (
    _clear_load_cache_for_tests,
    _load_to_device,
    apply_descriptor,
)

CPU = torch.device("cpu")


class Patcher:
    """Duck-types the two attributes ``_load_to_device`` reads."""

    def __init__(self, device=CPU):
        self.load_device = device


class ImageDescriptor:
    """A generic upscale descriptor (no ``_mambair_runner``) with an optional patcher."""

    scale = 2

    def __init__(self, patcher=None):
        self.patcher = patcher
        self.model = torch.nn.Identity()

    def __call__(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


@contextlib.contextmanager
def fake_model_management(calls, *, fail_times=0, device=CPU):
    """Install a fake ``comfy.model_management`` and record every load call.

    ``fail_times`` makes the first N calls raise, which is the transient-failure path.
    """
    state = {"fail": int(fail_times)}
    module = types.ModuleType("comfy.model_management")

    def load_models_gpu(patchers, **kwargs):
        calls.append((tuple(id(p) for p in patchers), kwargs))
        if state["fail"] > 0:
            state["fail"] -= 1
            raise RuntimeError("injected load failure")

    module.load_models_gpu = load_models_gpu
    module.intermediate_device = lambda: device

    package = types.ModuleType("comfy")
    package.__path__ = []
    package.model_management = module
    saved = {name: sys.modules.get(name) for name in ("comfy", "comfy.model_management")}
    sys.modules["comfy"] = package
    sys.modules["comfy.model_management"] = module
    try:
        yield calls
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _fresh():
    _clear_load_cache_for_tests()


# --------------------------------------------------------------------------- #
# REQ-5.1: one manager call per (model, device), not per tile
# --------------------------------------------------------------------------- #
def test_opt5_manager_runs_once_for_a_multi_tile_image_and_once_across_calls():
    _fresh()
    calls = []
    descriptor = ImageDescriptor(Patcher())
    image = torch.rand(1, 96, 96, 3)
    with fake_model_management(calls):
        # 96 px with a 64 px tile and 16 px overlap is a 2x2 grid: four tiles.
        first, _ = apply_descriptor(descriptor, image, tile=64, overlap=16)
        second, _ = apply_descriptor(descriptor, image, tile=64, overlap=16)
    assert first.shape == second.shape == (1, 192, 192, 3)
    assert len(calls) == 1, f"expected one manager call, got {len(calls)}"


def test_opt5_failed_load_is_not_cached_and_success_is():
    _fresh()
    calls = []
    descriptor = ImageDescriptor(Patcher())
    with fake_model_management(calls, fail_times=1):
        assert _load_to_device(descriptor) is None, "a failure must not report a device"
        assert _load_to_device(descriptor) == CPU
        assert _load_to_device(descriptor) == CPU  # served from the cache
    assert len(calls) == 2, f"expected a retry after the failure, got {len(calls)} calls"


# --------------------------------------------------------------------------- #
# REQ-5.3: the failure path still reloads (eviction recovery), ladder intact
# --------------------------------------------------------------------------- #
class EvictingDescriptor(ImageDescriptor):
    """Raises a device-mismatch error once, as ComfyUI's eviction does."""

    def __init__(self, patcher):
        super().__init__(patcher)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least two devices, "
                "cuda:0 and cpu!"
            )
        return super().__call__(x)


def test_opt5_device_mismatch_reloads_through_the_manager():
    _fresh()
    calls = []
    descriptor = EvictingDescriptor(Patcher())
    with fake_model_management(calls):
        out, _ = apply_descriptor(descriptor, torch.rand(1, 48, 48, 3), tile=64, overlap=16)
    assert out.shape == (1, 96, 96, 3)
    assert descriptor.calls == 2, "the forward must be retried once after a reload"
    assert len(calls) == 2, f"the reload must call the manager again, got {len(calls)}"


class _OOMOnce:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if self.calls == 1:
            raise torch.cuda.OutOfMemoryError("injected out-of-memory")
        return x


def test_opt5_oom_ladder_still_runs_its_reload_hook():
    _fresh()
    hooks = []
    fn = _OOMOnce()
    x = torch.rand(1, 3, 600, 600)
    out = tiling.bounded_forward(fn, x, 1, 512, 32, on_oom=lambda: hooks.append(1))
    assert out.shape == x.shape
    assert fn.calls > 2, "the ladder must re-run the whole image at the smaller cap"
    assert len(hooks) == 1, "one OOM retry must call the hook exactly once"


# --------------------------------------------------------------------------- #
# REQ-5.2: one clamp and one crossing to the output device
# --------------------------------------------------------------------------- #
class RecordingRunner:
    """Returns an out-of-range tensor and records the clamp contract."""

    hp = types.SimpleNamespace(family="recording")

    def __init__(self, device=CPU):
        self.clamp_args = []
        self.device = device

    def run(self, x, **kwargs):
        self.clamp_args.append(kwargs.get("clamp", "missing"))
        assert kwargs.get("on_oom") is not None, "apply must give the OOM path a reload hook"
        return torch.full(
            (x.shape[0], 3, x.shape[2] * 2, x.shape[3] * 2), 3.0, device=self.device
        )


class RunnerDescriptor(ImageDescriptor):
    def __init__(self, runner, patcher=None):
        super().__init__(patcher)
        self._mambair_runner = runner


def test_opt5_runner_does_not_clamp_and_apply_clamps_once():
    _fresh()
    runner = RecordingRunner()
    out, _ = apply_descriptor(RunnerDescriptor(runner), torch.rand(1, 32, 32, 3), tile=64)
    assert runner.clamp_args == [False], runner.clamp_args
    assert out.max().item() == 1.0 and out.min().item() == 1.0


def test_opt5_output_crosses_to_the_output_device_once():
    if not torch.cuda.is_available():
        return "skip"
    _fresh()
    original = torch.Tensor.to
    transfers = []

    def counting(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Count only a full-size RGB output moving off the compute device; the input transfer
        # and the alpha resample have different element counts.
        if result is not self and self.is_cuda and self.numel() == 3 * 128 * 128:
            transfers.append(result.device)
        return result

    from mambair import apply as apply_mod

    runner = RecordingRunner(torch.device("cuda"))
    descriptor = RunnerDescriptor(runner)
    image = torch.rand(1, 64, 64, 4)  # an alpha channel exercises the cat path too
    saved_output = apply_mod.output_device
    apply_mod.output_device = lambda: CPU
    torch.Tensor.to = counting
    try:
        out, _ = apply_descriptor(descriptor, image, tile=64, overlap=16)
    finally:
        torch.Tensor.to = original
        apply_mod.output_device = saved_output
    assert out.device.type == "cpu"
    assert out.shape == (1, 128, 128, 4)
    assert len(transfers) == 1, f"the output crossed {len(transfers)} times: {transfers}"
