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


"""P-INF-1 / P-DET-1: hyperparameters and family detection from the state dict."""

import torch

import corpus
from mambair import infer as inf


def _synth_v2(
    *, embed=174, in_chans=3, depths=(6, 6), heads=6, window=16, d_state=16,
    inner_rank=64, num_tokens=128, mlp=2.0, upsampler="pixelshuffle", upscale=4,
    light=False, convffn_k=5,
):
    sd = {
        "conv_first.weight": torch.zeros(embed, in_chans, 3, 3),
        "conv_first.bias": torch.zeros(embed),
        "conv_after_body.weight": torch.zeros(embed, embed, 3, 3),
        "norm.weight": torch.zeros(embed),
        # a 1-D index buffer of window**4 entries; the old (window**4, window**4) shape
        # allocated 17 GB and only passed on a machine with that much free RAM
        "relative_position_index_SA": torch.zeros(window ** 4),
    }
    hidden = int(embed * mlp)
    for i, depth in enumerate(depths):
        for j in range(depth):
            p = f"layers.{i}.residual_group.layers.{j}."
            sd[p + "assm.selectiveScan.A_logs"] = torch.zeros(hidden, d_state)
            sd[p + "assm.in_proj.0.weight"] = torch.zeros(hidden, embed, 1, 1)
            sd[p + "assm.embeddingB.weight"] = torch.zeros(num_tokens, inner_rank)
            sd[p + "win_mhsa.relative_position_bias_table"] = torch.zeros((2 * window - 1) ** 2, heads)
            if light:
                sd[p + "convffn1.sg.conv.weight"] = torch.zeros(hidden // 2, 1, 5, 5)
            else:
                sd[p + "convffn1.dwconv.depthwise_conv.0.weight"] = torch.zeros(hidden, 1, convffn_k, convffn_k)
    if upsampler == "pixelshuffle":
        sd["conv_before_upsample.0.weight"] = torch.zeros(64, embed, 3, 3)
        n = 0
        remaining = upscale
        while remaining > 1:
            r = 3 if remaining == 3 else 2
            sd[f"upsample.{n}.weight"] = torch.zeros(64 * r * r, 64, 3, 3)
            remaining //= r
            n += 2
    elif upsampler == "pixelshuffledirect":
        sd["upsample.0.weight"] = torch.zeros(in_chans * upscale ** 2, embed, 3, 3)
    elif upsampler == "nearest+conv":
        # upstream's real-world SR path has two x2 steps, so both convs exist
        sd["conv_up1.weight"] = torch.zeros(64, 64, 3, 3)
        sd["conv_up2.weight"] = torch.zeros(64, 64, 3, 3)
    return sd


def test_detect_v2():
    assert inf.detect_family(_synth_v2()) == "MambaIRv2"
    assert inf.detect_family({"params": _synth_v2()}) == "MambaIRv2"


def test_detect_v2_light():
    sd = _synth_v2(embed=48, depths=(5, 5), heads=4, mlp=1.0, upsampler="pixelshuffledirect",
                   inner_rank=32, num_tokens=64, light=True)
    assert inf.detect_family(sd) == "MambaIRv2Light"


def test_detect_rejects_foreign_architectures():
    foreign = [
        {"model.0.weight": torch.zeros(64, 3, 3, 3), "model.1.weight": torch.zeros(64)},
        {"layers.0.residual_group.blocks.0.norm1.weight": torch.zeros(180)},
        {"conv_first.weight": torch.zeros(180, 3, 3, 3)},
        {},
    ]
    for sd in foreign:
        assert inf.detect_family(sd) is None, sd.keys()


def test_infer_synthetic_v2():
    sd = _synth_v2(depths=(6, 6, 6, 6, 6, 6, 6, 6, 6), upscale=4)
    hp = inf.infer(sd)
    assert hp.family == "MambaIRv2"
    assert (hp.embed_dim, hp.in_chans, hp.upscale) == (174, 3, 4)
    assert hp.upsampler == "pixelshuffle"
    assert hp.depths == (6,) * 9
    assert hp.num_heads == (6,) * 9
    assert (hp.window_size, hp.d_state, hp.inner_rank, hp.num_tokens) == (16, 16, 64, 128)
    assert hp.mlp_ratio == 2.0 and hp.convffn_kernel_size == 5
    assert hp.purpose == "SR"


def test_infer_scales():
    for upscale, expect in ((2, 2), (3, 3), (4, 4)):
        hp = inf.infer(_synth_v2(depths=(2,), upscale=upscale))
        assert (hp.upscale, hp.upsampler) == (expect, "pixelshuffle"), (upscale, hp.upscale)

    direct = inf.infer(_synth_v2(depths=(2,), upsampler="pixelshuffledirect", upscale=4, embed=48, mlp=1.0))
    assert (direct.upscale, direct.upsampler) == (4, "pixelshuffledirect")

    real = inf.infer(_synth_v2(depths=(2,), upsampler="nearest+conv", upscale=4))
    assert (real.upscale, real.upsampler) == (4, "nearest+conv")

    restore = inf.infer(_synth_v2(depths=(2,), upsampler="", upscale=1))
    assert (restore.upscale, restore.upsampler, restore.purpose) == (1, "", "Restoration")


def test_infer_grayscale():
    hp = inf.infer(_synth_v2(depths=(2,), in_chans=1, upsampler="", upscale=1))
    assert hp.in_chans == 1 and hp.upscale == 1


def test_infer_module_prefix_and_wrapper():
    sd = _synth_v2(depths=(2,))
    assert inf.detect_family({"module." + k: v for k, v in sd.items()}) == "MambaIRv2"
    unwrapped = inf.canonicalize({"params_ema": sd})
    assert set(unwrapped) == set(sd), "wrapper key must be removed"


def test_infer_real_checkpoint_x4():
    name = corpus.V2["x4"]
    if not corpus.exists(name):
        return "skip"
    hp = inf.infer(corpus.load(name))
    for key, expected in corpus.V2_LARGE_X4.items():
        actual = getattr(hp, key)
        assert actual == expected, (key, actual, expected)


def _synth_v1(*, embed=180, in_chans=3, depths=(6, 6), d_state=16, mlp=2.0,
              upsampler="pixelshuffle", upscale=4, attn_name="self_attention"):
    sd = {
        "conv_first.weight": torch.zeros(embed, in_chans, 3, 3),
        "conv_first.bias": torch.zeros(embed),
        "conv_after_body.weight": torch.zeros(embed, embed, 3, 3),
        "norm.weight": torch.zeros(embed),
    }
    hidden = int(embed * mlp)
    for i in range(len(depths)):
        for j in range(depths[i]):
            p = f"layers.{i}.residual_group.blocks.{j}."
            sd[p + f"{attn_name}.x_proj_weight"] = torch.zeros(4, 44, hidden)
            sd[p + f"{attn_name}.A_logs"] = torch.zeros(4 * hidden, d_state)
            sd[p + f"{attn_name}.conv2d.weight"] = torch.zeros(hidden, 1, 3, 3)
            sd[p + "conv_blk.cab.0.weight"] = torch.zeros(embed // 3, embed, 3, 3)
    if upsampler == "pixelshuffle":
        sd["conv_before_upsample.0.weight"] = torch.zeros(64, embed, 3, 3)
        n, remaining = 0, upscale
        while remaining > 1:
            r = 3 if remaining == 3 else 2
            sd[f"upsample.{n}.weight"] = torch.zeros(64 * r * r, 64, 3, 3)
            remaining //= r
            n += 2
        sd["conv_last.weight"] = torch.zeros(in_chans, 64, 3, 3)
    elif upsampler == "pixelshuffledirect":
        sd["upsample.0.weight"] = torch.zeros(in_chans * upscale ** 2, embed, 3, 3)
    elif upsampler == "nearest+conv":
        # upstream's real-world SR path has two x2 steps, so both convs exist
        sd["conv_up1.weight"] = torch.zeros(64, 64, 3, 3)
        sd["conv_up2.weight"] = torch.zeros(64, 64, 3, 3)
    else:
        sd["conv_last.weight"] = torch.zeros(in_chans, embed, 3, 3)
    return sd


def test_detect_v1():
    assert inf.detect_family(_synth_v1()) == "MambaIR"
    assert inf.detect_family(_synth_v1(attn_name="attn")) == "MambaIR"


def test_infer_v1():
    hp = inf.infer(_synth_v1(depths=(6, 6, 6, 6, 6, 6), mlp=1.2))
    assert (hp.family, hp.embed_dim, hp.in_chans, hp.upscale) == ("MambaIR", 180, 3, 4)
    assert hp.depths == (6,) * 6
    assert hp.mlp_ratio == 1.2 and hp.d_state == 16
    assert hp.upsampler == "pixelshuffle" and hp.purpose == "SR"

    light = inf.infer(_synth_v1(upsampler="pixelshuffledirect", upscale=2))
    assert (light.upscale, light.upsampler) == (2, "pixelshuffledirect")

    # `nearest+conv` exists in MambaIRv2 but not in v1 upstream (verified: the
    # `mambair` branch has no conv_up/conv_hr keys). Rejecting it here is what stops
    # a fork checkpoint from silently falling into the denoising branch and
    # reporting x4 while behaving as x1 (hardening spec BUG-6).
    try:
        inf.infer(_synth_v1(upsampler="nearest+conv", upscale=4))
    except inf.UnsupportedUpsampler:
        pass
    else:
        raise AssertionError("v1 must reject nearest+conv")

    dn = inf.infer(_synth_v1(upsampler="", upscale=1, in_chans=1))
    assert (dn.upscale, dn.upsampler, dn.purpose, dn.in_chans) == (1, "", "Restoration", 1)



def test_infer_real_v1_checkpoints():
    x4 = corpus.V1["sr_x4"]
    if corpus.exists(x4):
        hp = inf.infer(corpus.load(x4))
        assert (hp.family, hp.upscale, hp.embed_dim, hp.in_chans) == ("MambaIR", 4, 180, 3)
        assert hp.depths == (6,) * 6
        assert hp.mlp_ratio == 2.0 and hp.d_state == 16
        assert hp.upsampler == "pixelshuffle"
    else:
        return "skip"

    car = corpus.V1["car_q10"]
    if corpus.exists(car):
        hp = inf.infer(corpus.load(car))
        assert (hp.family, hp.upscale, hp.in_chans, hp.upsampler) == ("MambaIR", 1, 1, "")
        assert hp.depths == (6,) * 6 and hp.mlp_ratio == 1.2
        assert hp.purpose == "Restoration"


def test_infer_real_v2_light():
    name = corpus.V2LIGHT["x2"]
    if not corpus.exists(name):
        return "skip"
    hp = inf.infer(corpus.load(name))
    assert (hp.family, hp.upscale, hp.embed_dim) == ("MambaIRv2Light", 2, 48)
    assert hp.depths == (5, 5, 5, 5)
    assert (hp.d_state, hp.inner_rank, hp.num_tokens) == (8, 32, 64)
    assert hp.upsampler == "pixelshuffledirect"


def _synth_unet(*, dim=48, in_chans=3, out_chans=3, blocks=(4, 6, 6, 8), mlp=1.5,
                d_state=16, refinement=4, bias=False):
    sd = {
        "patch_embed.proj.weight": torch.zeros(dim, in_chans, 3, 3),
        "output.weight": torch.zeros(out_chans, dim * 2, 3, 3),
        "down1_2.body.0.weight": torch.zeros(dim // 2, dim, 3, 3),
        "up2_1.body.0.weight": torch.zeros(dim * 2, dim * 2, 3, 3),
    }
    if bias:
        sd["output.bias"] = torch.zeros(out_chans)
    hidden = int(dim * mlp)
    for group, count in zip(("encoder_level1", "encoder_level2", "encoder_level3", "latent"), blocks):
        for i in range(count):
            p = f"{group}.{i}.self_attention."
            sd[p + "x_proj_weight"] = torch.zeros(4, 35, hidden)
            sd[p + "A_logs"] = torch.zeros(4 * hidden, d_state)
    for group in ("decoder_level3", "decoder_level2", "decoder_level1"):
        sd[group + ".0.self_attention.x_proj_weight"] = torch.zeros(4, 35, hidden)
        sd[group + ".0.self_attention.A_logs"] = torch.zeros(4 * hidden, d_state)
    for i in range(refinement):
        sd[f"refinement.{i}.self_attention.x_proj_weight"] = torch.zeros(4, 35, hidden)
    return sd


def test_detect_unet():
    assert inf.detect_family(_synth_unet()) == "MambaIRUNet"


def test_infer_unet():
    hp = inf.infer(_synth_unet())
    assert (hp.family, hp.embed_dim, hp.in_chans, hp.out_chans) == ("MambaIRUNet", 48, 3, 3)
    assert hp.depths == (4, 6, 6, 8)
    assert hp.num_refinement_blocks == 4
    assert hp.mlp_ratio == 1.5 and hp.d_state == 16
    assert (hp.upscale, hp.upsampler, hp.purpose) == (1, "", "Restoration")

    biased = inf.infer(_synth_unet(bias=True, in_chans=6, out_chans=3, blocks=(2,)))
    assert biased.bias is True and biased.in_chans == 6


def test_infer_real_unet():
    name = corpus.UNET["realDN"]
    if not corpus.exists(name):
        return "skip"
    hp = inf.infer(corpus.load(name))
    assert (hp.family, hp.embed_dim, hp.in_chans, hp.out_chans) == ("MambaIRUNet", 48, 3, 3)
    assert hp.depths == (4, 6, 6, 8)
    assert hp.mlp_ratio == 1.5 and hp.d_state == 16
    assert hp.num_refinement_blocks == 4 and hp.bias is False
    assert (hp.upscale, hp.purpose) == (1, "Restoration")
