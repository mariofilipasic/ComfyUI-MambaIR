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


"""T6.4 / REQ-4.7 / REQ-8.4: precision and device guards."""

import torch

import corpus


def test_cpu_forward_produces_a_valid_image():
    runner = corpus.tiny_runner()
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    y = runner.run(x, seed=0)
    assert y.shape == (1, 3, 32, 32), y.shape
    assert y.dtype == torch.float32
    assert y.min() >= 0 and y.max() <= 1 and not torch.isnan(y).any()


def test_bf16_on_cpu_falls_back_to_fp32():
    """Autocast bf16 is only for GPU; on CPU the request must be a no-op, not an error."""
    fp32 = corpus.tiny_runner(precision="fp32")
    bf16 = corpus.tiny_runner(precision="bf16")
    bf16.arch.load_state_dict(fp32.arch.state_dict())
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    a = fp32.run(x, seed=0)
    b = bf16.run(x, seed=0)
    assert b.dtype == torch.float32
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_bf16_on_cuda_stays_finite_and_close():
    if not torch.cuda.is_available():
        return "skip"
    fp32 = corpus.tiny_runner(precision="fp32")
    bf16 = corpus.tiny_runner(precision="bf16")
    fp32.arch.to("cuda")
    bf16.arch.to("cuda")
    bf16.arch.load_state_dict(fp32.arch.state_dict())
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.rand(1, 3, 16, 16, generator=g, device="cuda")
    with torch.inference_mode():
        a = fp32.run(x, seed=0, tile=64)
        b = bf16.run(x, seed=0, tile=64)
    assert b.dtype == torch.float32
    assert torch.isfinite(b).all()
    assert b.min() >= 0 and b.max() <= 1
    # bf16 convolutions change the result slightly; the scan itself stays fp32
    assert (a - b).abs().mean().item() < 0.02, (a - b).abs().mean().item()


def test_precision_setting_does_not_leak_between_runs():
    runner = corpus.tiny_runner(precision="fp32")
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(3))
    before = runner.run(x, seed=0)
    runner.run(x, seed=0, precision="bf16")  # per-call override
    after = runner.run(x, seed=0)
    assert torch.equal(before, after), "a per-call precision override must not stick"
    assert runner.precision == "fp32"
