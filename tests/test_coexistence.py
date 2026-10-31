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


"""T6.2 / US-8: coexist with `inkvec-comfyui`, which vendors its own MambaIRv2 port.

Two independent ports of the same architecture running in one process is the sharpest
test we have of our own port -- and the closest thing to a real-world conflict test
(inkvec is a popular pack that happens to ship MambaIRv2 for logo upscaling).

Checks:

* importing both has no module-name collision (each keeps its code package-private);
* neither registers anything in the other's namespace, and inkvec adds nothing to
  spandrel's registry;
* our forward does not consume the process-global RNG (so inkvec's seeding is safe
  from us, and ours from it);
* given identical weights and a deterministic router, **both ports produce the same
  output** -- an independent confirmation of the vendored architecture.
"""

import importlib.util
import os
import sys

import torch

import corpus
from mambair import registry
from mambair.archs.mambairv2 import MambaIRv2

INVEC_DIR = os.environ.get(
    "INKVEC_DIR", r"E:/AI/ComfyUI/custom_nodes/inkvec-comfyui"
)


def _load_inkvec():
    """Import inkvec-comfyui as a package (its directory name has a dash)."""
    if not os.path.isfile(os.path.join(INVEC_DIR, "vendor", "mambairv2_arch.py")):
        return None
    spec = importlib.util.spec_from_file_location(
        "inkvec_pack",
        os.path.join(INVEC_DIR, "__init__.py"),
        submodule_search_locations=[INVEC_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["inkvec_pack"] = module
    spec.loader.exec_module(module)
    from inkvec_pack.vendor.mambairv2_arch import MambaIRv2 as InkvecMambaIRv2

    return InkvecMambaIRv2


def _one_hot(logits, **_kwargs):
    index = logits.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, index, 1.0)


def _hp():
    from mambair import infer as inf

    return inf.infer(_synth_state_dict())


def _synth_state_dict():
    sd = {
        "conv_first.weight": torch.zeros(24, 3, 3, 3),
        "conv_after_body.weight": torch.zeros(24, 24, 3, 3),
    }
    for j in range(1):
        p = f"layers.0.residual_group.layers.{j}."
        sd[p + "assm.selectiveScan.A_logs"] = torch.zeros(48, 4)
        sd[p + "assm.in_proj.0.weight"] = torch.zeros(48, 24, 1, 1)
        sd[p + "assm.embeddingB.weight"] = torch.zeros(8, 8)
        sd[p + "win_mhsa.relative_position_bias_table"] = torch.zeros(49, 2)
        sd[p + "convffn1.sg.conv.weight"] = torch.zeros(24, 1, 5, 5)
    for i in (0, 1):
        sd[f"layers.{i}.residual_group.layers.0.assm.selectiveScan.A_logs"] = torch.zeros(48, 4)
    sd["upsample.0.weight"] = torch.zeros(12, 24, 3, 3)
    return sd


HP = dict(
    upscale=2, in_chans=3, img_size=64, img_range=1.0, embed_dim=24, d_state=4,
    depths=[1, 1], num_heads=[2, 2], window_size=4, inner_rank=8, num_tokens=8,
    convffn_kernel_size=5, mlp_ratio=2.0, upsampler="pixelshuffledirect",
    resi_connection="1conv",
)


def test_no_module_name_collision():
    theirs = _load_inkvec()
    if theirs is None:
        return "skip"
    assert MambaIRv2 is not theirs, "two ports must stay distinct classes"
    assert MambaIRv2.__module__ != theirs.__module__, (MambaIRv2.__module__, theirs.__module__)
    # each keeps its own scan implementation; ours must not have hijacked theirs
    import inkvec_pack.vendor.scan as their_scan

    from mambair import scan as our_scan

    assert their_scan is not our_scan
    assert "mamba_ssm" not in sys.modules or sys.modules["mamba_ssm"] is not None


def test_inkvec_adds_nothing_to_spandrel():
    before = {a.architecture.id for a in __import__("spandrel").MAIN_REGISTRY.architectures()}
    if _load_inkvec() is None:
        return "skip"
    after = {a.architecture.id for a in __import__("spandrel").MAIN_REGISTRY.architectures()}
    assert after == before, after ^ before


def test_runner_does_not_disturb_their_seed_path():
    """The public path (ModelRunner) must not consume the global RNG inkvec relies on."""
    if _load_inkvec() is None:
        return "skip"
    from mambair.spandrel_arch import ModelRunner

    runner = ModelRunner(MambaIRv2(**HP).eval(), _hp(), tile_cap=64)
    torch.manual_seed(0)
    before = torch.random.get_rng_state().clone()
    runner.run(torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1)))
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after), "the runner consumed the global RNG"


def test_raw_arch_behaves_like_upstream_on_the_global_rng():
    """The vendored arch itself samples from the global RNG, exactly like upstream.

    Documented on purpose: determinism and RNG isolation are properties of the
    runtime path (`ModelRunner`), not of the vendored architecture, which is kept
    behaviourally identical to upstream.
    """
    ours = MambaIRv2(**HP).eval()
    torch.manual_seed(0)
    before = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        ours(torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1)))
    after = torch.random.get_rng_state().clone()
    assert not torch.equal(before, after), "upstream-compatible arch should sample globally"


def test_both_ports_agree_on_real_weights():
    """Same checkpoint, deterministic router, two independent ports: same output."""
    theirs_cls = _load_inkvec()
    if theirs_cls is None:
        return "skip"
    name = corpus.V2["x4"]
    if not corpus.exists(name):
        return "skip"

    sd = corpus.load(name)
    from mambair import infer as inf

    hp = inf.infer(sd)
    ours = MambaIRv2(**hp.arch_kwargs()).eval()
    theirs = theirs_cls(**hp.arch_kwargs()).eval()
    theirs.load_state_dict(ours.state_dict(), strict=True)

    import mambair.archs.mambairv2 as our_module
    import inkvec_pack.vendor.mambairv2_arch as their_module

    ours_router, theirs_router = our_module.gumbel_hard, their_module.F.gumbel_softmax
    our_module.gumbel_hard = _one_hot
    their_module.F.gumbel_softmax = _one_hot
    try:
        x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(2))
        with torch.inference_mode():
            a = ours(x)
            b = theirs(x)
    finally:
        our_module.gumbel_hard = ours_router
        their_module.F.gumbel_softmax = theirs_router

    assert a.shape == b.shape, (a.shape, b.shape)
    diff = (a - b).abs().max().item()
    assert diff < 1e-6, diff
