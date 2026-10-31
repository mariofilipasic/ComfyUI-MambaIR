# Provenance

This pack contains vendored architecture code. Upstream is Apache-2.0; the
vendored files keep their licence and record their origin here (NFR-5).

The pack around them is licensed **AGPL-3.0-or-later** (`LICENSE`). The vendored
files are **not**: they stay Apache-2.0 (`LICENSE-APACHE-2.0`) and carry no AGPL
header, which `tests/test_license.py` asserts in both directions.

## Upstream source

| | |
|---|---|
| Project | MambaIR / MambaIRv2 — "A Simple Baseline for Image Restoration with State-Space Model" (ECCV24), "Attentive State Space Restoration" (CVPR25) |
| Authors | Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, Yawei Li |
| Repository | https://github.com/csguoh/MambaIR |
| Licence | Apache License 2.0 — see `LICENSE-APACHE-2.0` (this pack's own code is AGPL-3.0-or-later, `LICENSE`) |
| Local checkout | `G:/repo/comfyui/MambaIR` |
| Commit | `33d7b3460c4665229334cc8de38c7f4c766ed3be` (`refs/heads/main`) |
| Weights | https://github.com/csguoh/MambaIR/releases and https://huggingface.co/cguoh/MambaIR (not redistributed here) |

## Vendored files

| File here | Upstream file | Modifications |
|---|---|---|
| `mambair/archs/mambairv2.py` | `basicsr/archs/mambairv2_arch.py` | imports (local shims + `mambair.scan`), `gumbel_hard` hook, dropped demo block |
| `mambair/archs/mambairv2light.py` | `basicsr/archs/mambairv2light_arch.py` | same as above |
| `mambair/archs/mambair.py` | `basicsr/archs/mambair_arch.py` | imports only (upstream `main` and `mambair` branch are identical, 894 lines, 0 diff) |
| `mambair/archs/__init__.py` | `basicsr/archs/arch_util.py`, `timm.models.layers` | `to_2tuple`, `trunc_normal_`, `DropPath` re-implemented (no timm/basicsr dependency); Apache-2.0, so no AGPL header |

No architectural changes: `tests/test_parity.py` asserts that the vendored
`MambaIRv2` and `MambaIR` modules have an identical state-dict key set to the
unmodified upstream files, and that both produce identical outputs from identical
weights.

## Not vendored

The selective scan is **not** copied from `mamba_ssm`: `mambair/scan.py` is an
independent implementation of the same operator (associative chunked scan), with
the option to dispatch to `mamba_ssm`, to a Triton kernel, or to PyTorch.

## Third-party notices

* MambaIR (Apache-2.0) — https://github.com/csguoh/MambaIR
* MambaIRv2 builds on BasicSR (Apache-2.0), ART (Apache-2.0) and VMamba (Apache-2.0);
  none of those are vendored here, only the MambaIR sources listed above.
