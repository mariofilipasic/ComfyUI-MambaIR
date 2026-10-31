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


"""State dict -> family + constructor arguments.

The released MambaIR checkpoints are BasicSR-style ``{"params": ...}`` blobs with
**no hyperparameters** stored, and the families document no filename convention.
Everything below is therefore reconstructed from tensor shapes and key names.
Every row is covered by ``tests/test_infer.py``.

Pure functions only (no torch modules are constructed here), so this module is
cheap to unit-test and cheap to import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "Hyperparams",
    "UnsupportedUpsampler",
    "canonicalize",
    "detect_family",
    "infer",
    "FAMILIES",
    "SUPPORTED_FAMILIES",
]


class UnsupportedUpsampler(ValueError):
    """Raised when a checkpoint uses an upsampler that the matching architecture lacks.

    Carries the upsampler name in ``args[0]`` so callers can build their own message.
    """

FAMILIES = ("MambaIRv2", "MambaIRv2Light", "MambaIR", "MambaIRUNet")

#: Families with a registered spandrel architecture.
SUPPORTED_FAMILIES = ("MambaIRv2", "MambaIRv2Light", "MambaIR", "MambaIRUNet")


@dataclass(frozen=True)
class Hyperparams:
    """Everything the architecture constructors need, reconstructed from tensor shapes.

    The released checkpoints store no hyperparameters and follow no filename convention, so every
    value here is inferred (see :func:`infer`). Frozen: a model's shape is not a setting.

    Attributes:
        family: One of :data:`FAMILIES`; selects which class is built and which arguments apply.
        in_chans: Input channels of ``conv_first``; 1 for luma-only restoration checkpoints.
        embed_dim: Width of the body (``conv_first`` output channels).
        upscale: Output scale factor, 1 for restoration.
        upsampler: ``"pixelshuffle"``, ``"pixelshuffledirect"``, ``"nearest+conv"`` or ``""``
            (the denoising/restoration path). Must be implemented by ``family`` -- see
            :class:`UnsupportedUpsampler`.
        depths: Blocks per outer layer, one entry per outer layer.
        mlp_ratio: Hidden width of the feed-forward block, as a multiple of ``embed_dim``.
        d_state: State dimension of the selective scan (``A_logs`` last axis).
        num_heads: Attention heads per outer layer (v2 families only).
        window_size: Side of the square attention window (v2 families only).
        inner_rank: Rank of the low-rank ``B`` embedding; ``embeddingB`` is an
            ``nn.Embedding(num_tokens, inner_rank)``.
        num_tokens: Size of the routing token dictionary (v2 families only).
        convffn_kernel_size: Depthwise kernel of the ConvFFN block.
        resi_connection: ``"1conv"`` or ``"3conv"`` between outer layers.
        img_size: Nominal training resolution; only used to size position tables.
        img_range: Dynamic range the network expects (1.0 for these checkpoints).
        out_chans: Output channels (UNet family).
        num_refinement_blocks: Decoder refinement blocks (UNet family).
        bias: Whether the head has a bias term, inferred from the presence of ``output.bias``.
        dual_pixel_task: Two-output defocus deblurring (UNet family).
    """

    family: str
    in_chans: int
    embed_dim: int
    upscale: int
    upsampler: str
    depths: tuple[int, ...]
    mlp_ratio: float = 2.0
    d_state: int = 16
    num_heads: tuple[int, ...] = field(default_factory=tuple)
    window_size: int = 16
    inner_rank: int = 64
    num_tokens: int = 128
    convffn_kernel_size: int = 5
    resi_connection: str = "1conv"
    img_size: int = 64
    img_range: float = 1.0
    out_chans: int = 3
    num_refinement_blocks: int = 4
    bias: bool = False
    dual_pixel_task: bool = False

    @property
    def purpose(self) -> str:
        """spandrel's purpose string: ``"Restoration"`` at scale 1, else ``"SR"``."""
        return "Restoration" if self.upscale == 1 else "SR"

    def arch_kwargs(self) -> dict:
        """Return the constructor kwargs for this family's vendored architecture class.

    Returns:
        A fresh dict; the caller may mutate it.

    Raises:
        NotImplementedError: If ``family`` has no constructor mapping.
    """
        if self.family in ("MambaIRv2", "MambaIRv2Light"):
            return dict(
                upscale=self.upscale,
                in_chans=self.in_chans,
                img_size=self.img_size,
                img_range=self.img_range,
                embed_dim=self.embed_dim,
                d_state=self.d_state,
                depths=list(self.depths),
                num_heads=list(self.num_heads),
                window_size=self.window_size,
                inner_rank=self.inner_rank,
                num_tokens=self.num_tokens,
                convffn_kernel_size=self.convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                upsampler=self.upsampler,
                resi_connection=self.resi_connection,
            )
        if self.family == "MambaIRUNet":
            return dict(
                inp_channels=self.in_chans,
                out_channels=self.out_chans,
                dim=self.embed_dim,
                num_blocks=list(self.depths),
                mlp_ratio=self.mlp_ratio,
                num_refinement_blocks=self.num_refinement_blocks,
                bias=self.bias,
                dual_pixel_task=self.dual_pixel_task,
            )
        if self.family == "MambaIR":
            return dict(
                upscale=self.upscale,
                in_chans=self.in_chans,
                img_size=self.img_size,
                img_range=self.img_range,
                embed_dim=self.embed_dim,
                d_state=self.d_state,
                depths=list(self.depths),
                mlp_ratio=self.mlp_ratio,
                upsampler=self.upsampler,
                resi_connection=self.resi_connection,
            )
        raise NotImplementedError(f"family {self.family!r} is not implemented yet")


# --------------------------------------------------------------------------- #
# state dict handling
# --------------------------------------------------------------------------- #
_WRAPPERS = ("params_ema", "params-ema", "params", "state_dict", "model")


def canonicalize(sd) -> dict:
    """Unwrap ``params``/``params_ema`` and strip a ``module.`` prefix.

    spandrel's ``ModelLoader`` does this for the native path; doing it again is
    harmless and makes the functions here usable on raw checkpoints in tests.
    """
    if not isinstance(sd, dict):
        raise TypeError(f"expected a state dict, got {type(sd)!r}")
    for wrapper in _WRAPPERS:  # e.g. {'params': ..., 'params_ema': ...}
        if wrapper in sd and isinstance(sd[wrapper], dict):
            sd = sd[wrapper]
            break
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def _first(sd, prefix):
    """The first key starting with ``prefix``, or ``None``."""
    for k in sd:
        if k.startswith(prefix):
            return k
    return None


def _shape(sd, prefix):
    """Shape of the first tensor whose key starts with ``prefix``."""
    k = _first(sd, prefix)
    if k is None:
        raise KeyError(prefix)
    return tuple(sd[k].shape)


def _blocks_per_outer_layer(sd, marker: str) -> int:
    """Count blocks in the first outer layer's ``residual_group.<marker>.<i>``."""
    prefix = f"layers.0.residual_group.{marker}."
    idx = {k[len(prefix):].split(".", 1)[0] for k in sd if k.startswith(prefix)}
    return len(idx)


def _outer_layers(sd) -> int:
    """Number of ``layers.<i>`` outer layers in the body."""
    idx = {k.split(".", 2)[1] for k in sd if k.startswith("layers.")}
    return len(idx)


def detect_family(sd) -> str | None:
    """Identify which MambaIR family produced this state dict.

    Ordered most-specific-first. ``None`` means "not ours", which is what keeps this pack from
    claiming SwinIR/HAT/ESRGAN checkpoints.

    Args:
        sd: A state dict, wrapped or unwrapped.

    Returns:
        One of :data:`FAMILIES`, or ``None``.
    """
    if not isinstance(sd, dict) or not sd:
        return None
    sd = canonicalize(sd)
    keys = sd.keys()

    def any_key(*prefixes):
        return any(k.startswith(p) for p in prefixes for k in keys)

    # MambaIRUNet: encoder/decoder ladder
    if any_key("encoder_level1.0.", "encoder_level1.") and any_key("decoder_level3.") and any_key("output."):
        return "MambaIRUNet"

    block = "layers.0.residual_group.layers.0."
    has_v2_block = (block + "assm.selectiveScan.A_logs") in keys
    if has_v2_block:
        if any_key(block + "convffn1.sg.conv.weight"):
            return "MambaIRv2Light"  # GatedMLP gate
        if any_key(block + "convffn1.dwconv.depthwise_conv.0.weight", block + "convffn1.fc1.weight"):
            return "MambaIRv2"  # ConvFFN
        return None

    # MambaIR (v1): ResidualGroup.blocks[].self_attention (SS2D, K=4 cross scans).
    # Some exports name the module "attn"; accept both.
    if any_key("layers.0.residual_group.blocks.0.self_attention.",
               "layers.0.residual_group.blocks.0.attn."):
        return "MambaIR"
    return None


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _infer_upscaler(sd, in_chans: int, family: str) -> tuple[int, str]:
    """Recover ``(scale, upsampler)`` from the reconstruction-head tensors.

    Returns:
        The scale factor and the upsampler name accepted by the family's constructor.

    Raises:
        UnsupportedUpsampler: If the checkpoint needs an upsampler this family lacks.
        ValueError: If the head shape does not correspond to any supported upsampler.
    """
    has_before = _first(sd, "conv_before_upsample.") is not None
    stages = [k for k in sd if k.startswith("upsample.") and k.endswith(".weight")]

    if _first(sd, "conv_up1.") is not None:
        # Real-world SR: two nearest+conv x2 steps, so x4. Only the v2 families implement it;
        # the v1 architecture has no such branch and would silently fall into its denoising
        # path, reporting x4 while behaving as x1.
        if family in ("MambaIRv2", "MambaIRv2Light") and _first(sd, "conv_up2.") is not None:
            return 4, "nearest+conv"
        raise UnsupportedUpsampler("nearest+conv")

    if stages:
        ratios = []
        for k in stages:
            out_c, in_c = sd[k].shape[0], sd[k].shape[1]
            ratios.append(out_c / in_c)
        if has_before:  # pixelshuffle: one conv per 2x or 3x step
            scale = 1
            for r in ratios:
                root = math.isqrt(int(round(r)))
                if root * root != int(round(r)):
                    raise ValueError(f"non-square pixelshuffle stage ratio {r}")
                scale *= root
            return scale, "pixelshuffle"
        # pixelshuffledirect: one conv producing scale^2 * out_channels
        if len(ratios) != 1:
            raise ValueError(f"pixelshuffledirect expects one stage, got {len(stages)}")
        total = sd[stages[0]].shape[0]
        if total % in_chans:
            raise ValueError(f"cannot infer direct-upscale factor from {total} / {in_chans}")
        root = math.isqrt(total // in_chans)
        if root * root != total // in_chans:
            raise ValueError("cannot infer direct-upscale factor")
        return root, "pixelshuffledirect"

    return 1, ""


def _infer_mlp_ratio(sd, embed_dim: int, family: str) -> float:
    """Feed-forward hidden width as a multiple of ``embed_dim``."""
    if family in ("MambaIRv2", "MambaIRv2Light"):
        hidden = _shape(sd, "layers.0.residual_group.layers.0.assm.in_proj.0.weight")[0]
        return hidden / embed_dim
    # v1: SS2D inner width is the last dim of the stacked x_proj weights (K, rank+2N, inner)
    hidden = _shape(sd, _v1_block(sd) + "x_proj_weight")[-1]
    return hidden / embed_dim


def _v1_block(sd) -> str:
    """Prefix of the first v1 block, tolerating the ``attn`` / ``self_attention`` naming."""
    for name in ("self_attention.", "attn."):
        prefix = "layers.0.residual_group.blocks.0."
        if (prefix + name + "x_proj_weight") in sd:
            return prefix + name
    raise KeyError("layers.0.residual_group.blocks.0.{self_attention,attn}.x_proj_weight")


def infer(sd, family: str | None = None) -> Hyperparams:
    """Reconstruct the hyperparameters of a MambaIR checkpoint.

    Args:
        sd: A state dict, optionally wrapped in ``{"params": ...}`` and/or carrying ``module.``
            prefixes.
        family: Force a family instead of detecting it.

    Returns:
        The inferred :class:`Hyperparams`.

    Raises:
        ValueError: If the state dict does not look like a MambaIR checkpoint.
        UnsupportedUpsampler: If the checkpoint needs an upsampler its family does not implement.
        NotImplementedError: If the family is recognised but has no loader in this pack.
        KeyError: If a required tensor is missing from an otherwise plausible checkpoint.
    """
    sd = canonicalize(sd)
    family = family or detect_family(sd)
    if family is None:
        raise ValueError("state dict does not look like a MambaIR checkpoint")
    if family not in SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"{family} checkpoints are recognised but not loadable yet (implemented: "
            f"{', '.join(SUPPORTED_FAMILIES)})"
        )

    if family == "MambaIRUNet":
        proj = _shape(sd, "patch_embed.proj.weight")
        embed_dim, in_chans = proj[0], proj[1]
        out_chans = _shape(sd, "output.weight")[0]
        mlp_ratio = _shape(sd, "encoder_level1.0.self_attention.x_proj_weight")[-1] / embed_dim
        d_state = _shape(sd, "encoder_level1.0.self_attention.A_logs")[-1]
        num_blocks = tuple(
            len({k.split(".", 2)[1] for k in sd if k.startswith(group + ".")})
            for group in ("encoder_level1", "encoder_level2", "encoder_level3", "latent")
        )
        return Hyperparams(
            family=family,
            in_chans=in_chans,
            embed_dim=embed_dim,
            upscale=1,
            upsampler="",
            depths=num_blocks,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            out_chans=out_chans,
            num_refinement_blocks=len({k.split(".", 2)[1] for k in sd if k.startswith("refinement.")}),
            bias=any(k == "output.bias" for k in sd),
        )

    conv_first = _shape(sd, "conv_first.weight")
    embed_dim, in_chans = conv_first[0], conv_first[1]
    upscale, upsampler = _infer_upscaler(sd, in_chans, family)

    if family == "MambaIR":
        depths = tuple(
            _blocks_per_outer_layer(sd, "blocks") for _ in range(_outer_layers(sd))
        )
        mlp_ratio = _infer_mlp_ratio(sd, embed_dim, family)
        block = _v1_block(sd)
        d_state = _shape(sd, block + "A_logs")[-1]
        resi = "3conv" if _first(sd, "conv_after_body.0.") is not None else "1conv"
        return Hyperparams(
            family=family,
            in_chans=in_chans,
            embed_dim=embed_dim,
            upscale=upscale,
            upsampler=upsampler,
            depths=depths,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            resi_connection=resi,
        )

    depths = tuple(_blocks_per_outer_layer(sd, "layers") for _ in range(_outer_layers(sd)))

    table = _shape(sd, "layers.0.residual_group.layers.0.win_mhsa.relative_position_bias_table")
    window = (math.isqrt(table[0]) + 1) // 2
    num_heads = table[1]

    block = "layers.0.residual_group.layers.0."
    d_state = _shape(sd, block + "assm.selectiveScan.A_logs")[1]
    embedding_b = _shape(sd, block + "assm.embeddingB.weight")

    if family == "MambaIRv2":
        convffn_k = _shape(sd, block + "convffn1.dwconv.depthwise_conv.0.weight")[-1]
    else:
        convffn_k = 5  # GatedMLP's Gate conv is hard-coded kernel_size=5 upstream

    resi = "3conv" if _first(sd, "conv_after_body.0.") is not None else "1conv"

    return Hyperparams(
        family=family,
        in_chans=in_chans,
        embed_dim=embed_dim,
        upscale=upscale,
        upsampler=upsampler,
        depths=depths,
        mlp_ratio=_infer_mlp_ratio(sd, embed_dim, family),
        d_state=d_state,
        num_heads=(num_heads,) * len(depths),
        window_size=window,
        inner_rank=embedding_b[1],
        num_tokens=embedding_b[0],
        convffn_kernel_size=convffn_k,
        resi_connection=resi,
    )
