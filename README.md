# ComfyUI-MambaIR

[MambaIR](https://github.com/csguoh/MambaIR) (ECCV24) and MambaIRv2 (CVPR25) — state-space-model
image restoration — as **native ComfyUI upscale models**.

Install it, drop a checkpoint into `models/upscale_models/`, and the ordinary three nodes just work:

```
Load Upscale Model (mambairv2_classicSR_Large_x4.pth) → Upscale Image (using Model) → Preview Image
```

No custom loader, no fixed scale, no separate pipeline: the pack registers the architectures with
[spandrel](https://github.com/chaiNNer-org/spandrel), which is what ComfyUI's own loader uses.

- [Install](#install) · [Models](#models) · [The optional node](#the-optional-node-mambair-upscale-image)
- [Performance](#performance) · [Notes](#notes) · [Benchmark](#benchmark)
- [Troubleshooting](#troubleshooting) · [Development](#development)
- [Licence and what it obliges you to](#licence-and-what-it-obliges-you-to)

## Install

ComfyUI Manager → search "MambaIR", or:

```sh
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-MambaIR
# restart ComfyUI — there are no Python dependencies to install
```

Requirements: a recent ComfyUI (V3 node API, ≥ v0.3.48; older builds get a V1 node automatically),
`torch`, `einops`, `spandrel` — all shipped with ComfyUI. **No CUDA toolchain, no `mamba_ssm`,
no `causal_conv1d`.** Developed and measured on CUDA and CPU; the scan is plain PyTorch, so other
backends should work but are unverified.

## Models

Weights are not bundled. Download from the
[upstream releases](https://github.com/csguoh/MambaIR/releases) or
[HuggingFace](https://huggingface.co/cguoh/MambaIR) and put them in `ComfyUI/models/upscale_models/`.
The architecture, scale and width are read from the checkpoint itself — no config file, no filename
convention.

| Family | Checkpoints | Scale | Status |
|---|---|---|---|
| `MambaIRv2` | classic SR Base/Small/Large x2/x3/x4, light SR x2/x3/x4, Gaussian denoise, JPEG-CAR | 1–4 | ✅ |
| `MambaIR` (v1) | classic SR x2/x3/x4, light SR, Gaussian denoise, grayscale JPEG-CAR, real-world SR | 1–4 | ✅ |
| `MambaIRUNet` | real-world denoise (SIDD) | 1 | ✅ (tested with `realDN.pth`) |

Scale-1 checkpoints (denoisers, JPEG-artifact reduction) work through the same nodes and return an
image of the input size. Grayscale (1-channel) checkpoints are applied to the **luma** of an RGB
image; the input's chroma is kept.

Rename files however you like — the dropdown shows filenames, the loader ignores them.

## The optional node: `MambaIR Upscale Image`

Only needed for control the native node does not expose. It works with **any** upscale model.

| Widget | Meaning |
|---|---|
| `tile` | Internal tile edge in px. `0` = auto (320 on CUDA, 256 on CPU); values below 128 are raised to 128 and logged |
| `overlap` | Feathered overlap between tiles, ≥16 px |
| `seed` | `MambaIRv2` routes pixels with Gumbel sampling; a fixed seed makes runs reproducible. `-1` randomises each run |
| `precision` | `bf16` speeds up convolutions on CUDA; the scan always stays fp32 |
| `verbose` | Print backend, tile and timing to the console |

It also outputs a `status` string, e.g. `MambaIRv2 x4 via mambair | 1×512×512 -> 2048×2048 | 61.4s`
(on a 6 GB laptop GPU — the timing is a property of your machine, not a promise).
The scale is always the model's own and is reported there; there is no scale widget.

## Performance

**Memory.** The selective scan is global *per tile*, so tile size is a memory dial, and above 320 px
the limit is the network's own activation maps rather than the scan: one 320² × 348 fp32 feature map
is 142 MB and several are live through nine layers. The PyTorch tier discretises the scan chunk by
chunk (256 tokens), so its scratch does not grow with the tile, and the Triton tier allocates only
the output. The pack therefore bounds every forward pass at an internal cap (320 px on a CUDA
device, 256 px on the CPU) and feathers the seams, so ComfyUI's own "halve the tile on OOM" retry
never has to trigger.

**Speed.** Three tiers, tried in this order:

* **`mamba_ssm`** — used automatically when it is importable. It does not need to be, and there are
  no prebuilt wheels for recent Python/torch/CUDA combinations on Windows, so most installs never
  see it.
* **Triton** — opt-in with `pip install triton-windows` (52 MB, touches nothing else). One program
  per `(batch, channel)` row prefix-scans a block of tokens at a time with an associative scan, so
  neither `(batch, channels, tokens, state)` tensors nor per-chunk carry arrays are ever built. It
  is probed on first use for correctness *and* speed, and adopted only if it wins; the probe runs on
  deterministic synthetic tensors, so the verdict is a property of the machine and kernel rather
  than of the image loaded first. It beats the PyTorch tier at every sequence length: **5.9× at a
  128 px tile and 4.8× at 256–320 px** for MambaIRv2 Large, **24× and 22×** for the MambaIR v1
  family, whose four directional scans per block make the scan a much larger share of a tile.
* **pure PyTorch** (always available) — a chunked associative scan.

Triton and the PyTorch tier associate the scan differently, so they are not bit-identical: on a
320 px MambaIRv2 Large tile their clamped outputs differ by at most 0.013 (mean 0.0001), about a
tenth of the worst bf16 delta below. Both tiers remain deterministic for a fixed seed.

**Precision.** Everything above is fp32. The node's `precision` widget can run the non-scan path in
bf16 (the scan itself always stays fp32): **1.21× on a MambaIRv2 Large tile** (in the cap sweep,
9.97 s → 8.23 s at 320 px, peak VRAM 2600 → 2185 MB) and 1.12× on the v1 family (4.93 s → 4.41 s).
It changes pixels — on an 8-bit scale the mean difference is 0.003 and the worst is 0.11 — so it is
opt-in, not the default.

**Tile size.** The scan is global per tile, so the tile is both a speed and a context (quality) dial.
The cap sweep below is on a 6 GB card with the Triton tier, per output megapixel. The pack defaults
to **320 px**.

| family | 128 px | 256 px | 320 px | 384 px | 448 px |
|---|---|---|---|---|---|
| MambaIRv2 Large | 8.7 s | 8.1 s | **7.5 s** | 7.4 s | 34.3 s |
| MambaIRv2 Light | 4.3 s | 3.0 s | 2.8 s | 2.8 s | **2.7 s** |
| MambaIR v1 | 4.7 s | 3.9 s | **3.7 s** | 17.7 s | 22.6 s |
| MambaIRUNet | 112 s | 38.5 s | 36.4 s | **16.0 s** | 15.7 s |

320 px stays the default because it is the largest cap that is fast for **all** families at once: the
v1 and v2 Large tiles collapse at 384 px (their peak allocation approaches the 6 GB card's usable
VRAM), and 448 px is worse still. The one family-specific exception is `MambaIRUNet`: at 384 px its
quarter-resolution scan crosses the fused-tier threshold (a 96×96 level is 9,216 tokens, above the
8,192-token floor; at 320 px the 80×80 level is 6,400 and falls back to the PyTorch scan), so it is
**2.3× faster at 384 px** (2.27 GB peak) and is worth setting by hand. That is a *context* change as
well as a speed one — the scan sees a 96×96 level instead of 80×80 — so it is a recommendation, not
the default.

Each tile re-reads an overlapping margin, so the work per useful pixel is `tiles × tile² / (H × W)`.
The last tile is flush with the edge, so the overhead is lumpy: a 320×320 image at a 256 px cap is
four tiles, **2.56× the work**, and takes **10.0 s at a 320 px cap (one tile) but 25.9 s at 256 px**.
The asymptotic `(tile/(tile-overlap))²` is 1.31× at 256/32, reached only once the grid is many tiles
wide.

Peak VRAM per tile (6 GB card): 256 px ≈ 1.7 GB, 320 px ≈ 2.6 GB (v1 ≈ 3.5 GB). Above 320 px the
limit is the network's own activation maps rather than the scan, so tiles much larger than that need
more VRAM than a 6 GB card has.

## Notes

**Reproducibility.** MambaIRv2's routing samples noise even in `eval()`: two runs on the same image
differ by 0.04 on average and by 0.44 on the worst pixel. The pack installs a local, seeded RNG for
the routing, so the same model + image + seed is byte-identical, and the process-global RNG is never
touched.

**Quality vs. upstream evaluation.** Restoring a large image tile-by-tile is *not* identical to the
paper's whole-image evaluation: the scan sees one tile's context. This is inherent to bounded tiling,
and is why the cap is configurable.

## Benchmark

**The short version:** install Triton (`pip install triton-windows`) — it is 5.9× faster than the
PyTorch scan on MambaIRv2 Large and 24× faster on the MambaIR v1 family — and keep the default
320 px cap, the largest tile that is fast for every family at once. The tables below let you check
that against your own card.

Each cap has two rows, one per tier: the gap between them is the speedup you are buying, and
`ms per output Mpx` is the column that compares across families. Every number comes from
`python tests/bench.py`, which measures the machine it runs on and splices its tables here
(`--write`); if you installed from source, re-run it after a driver or torch upgrade rather than
trusting the numbers below.

One table per family, because the families do not share a bottleneck: MambaIRv2 Large spends most of
a tile on elementwise traffic, the MambaIR v1 family on its four-direction scan. `scan GSU/s` is an
absolute rate (scan state-updates per second), so rows in different tables are comparable.

<!-- bench:start -->
Measured on **NVIDIA GeForce RTX 3060 Laptop GPU**, `mambairv2_classicSR_Large_x4.pth` (MambaIRv2 Large, x4), fp32, overlap 32, one tile per measurement (`python tests/bench.py --families v2`).

| cap | tokens | backend | forward | peak VRAM | scan GSU/s | ms per output Mpx |
|---|---|---|---|---|---|---|
| 128 px | 16,384 | triton (fused) | 1.29 s | 536 MB | 0.24 | 8731 |
| 128 px | 16,384 | torch | 7.62 s | 540 MB | 0.04 | 51672 |
| 176 px | 30,976 | triton (fused) | 2.70 s | 887 MB | 0.22 | 8150 |
| 176 px | 30,976 | torch | 14.24 s | 887 MB | 0.04 | 42908 |
| 256 px | 65,536 | triton (fused) | 6.48 s | 1717 MB | 0.19 | 8076 |
| 256 px | 65,536 | torch | 31.59 s | 1717 MB | 0.04 | 39346 |
| 320 px | 102,400 | triton (fused) | 10.04 s | 2600 MB | 0.19 | 7566 |
| 320 px | 102,400 | torch | 48.19 s | 2600 MB | 0.04 | 36309 |

Measured on **NVIDIA GeForce RTX 3060 Laptop GPU**, `mambairv2_lightSR_x2.pth` (MambaIRv2 Light, x2), fp32, overlap 32, one tile per measurement (`python tests/bench.py --families v2light`).

| cap | tokens | backend | forward | peak VRAM | scan GSU/s | ms per output Mpx |
|---|---|---|---|---|---|---|
| 128 px | 16,384 | triton (fused) | 0.16 s | 209 MB | 0.10 | 4332 |
| 128 px | 16,384 | torch | 2.05 s | 209 MB | 0.01 | 55728 |
| 176 px | 30,976 | triton (fused) | 0.30 s | 388 MB | 0.10 | 3631 |
| 176 px | 30,976 | torch | 3.64 s | 388 MB | 0.01 | 43895 |
| 256 px | 65,536 | triton (fused) | 0.60 s | 795 MB | 0.11 | 2971 |
| 256 px | 65,536 | torch | 7.29 s | 795 MB | 0.01 | 36334 |
| 320 px | 102,400 | triton (fused) | 0.94 s | 1234 MB | 0.10 | 2835 |
| 320 px | 102,400 | torch | 10.68 s | 1234 MB | 0.01 | 32194 |

Measured on **NVIDIA GeForce RTX 3060 Laptop GPU**, `MambaIR-real.pth` (MambaIR v1, x4, 4-direction scan), fp32, overlap 32, one tile per measurement (`python tests/bench.py --families v1`).

| cap | tokens | backend | forward | peak VRAM | scan GSU/s | ms per output Mpx |
|---|---|---|---|---|---|---|
| 128 px | 16,384 | triton (fused) | 0.69 s | 634 MB | 1.22 | 4708 |
| 128 px | 16,384 | torch | 16.49 s | 815 MB | 0.05 | 111843 |
| 176 px | 30,976 | triton (fused) | 1.39 s | 1115 MB | 1.15 | 4200 |
| 176 px | 30,976 | torch | 31.26 s | 1377 MB | 0.05 | 94208 |
| 256 px | 65,536 | triton (fused) | 3.10 s | 2253 MB | 1.10 | 3858 |
| 256 px | 65,536 | torch | 66.35 s | 2705 MB | 0.05 | 82645 |
| 320 px | 102,400 | triton (fused) | 4.90 s | 3465 MB | 1.08 | 3690 |
| 320 px | 102,400 | torch | 107.90 s | 4118 MB | 0.05 | 81302 |

Measured on **NVIDIA GeForce RTX 3060 Laptop GPU**, `realDN.pth` (MambaIRUNet, x1 denoise), fp32, overlap 32, one tile per measurement (`python tests/bench.py --families unet`).

| cap | tokens | backend | forward | peak VRAM | scan GSU/s | ms per output Mpx |
|---|---|---|---|---|---|---|
| 128 px | 16,384 | triton (fused) | 1.04 s | 352 MB | 0.14 | 112470 |
| 128 px | 16,384 | torch | 2.91 s | 424 MB | 0.05 | 315310 |
| 176 px | 30,976 | triton (fused) | 1.96 s | 565 MB | 0.14 | 94312 |
| 176 px | 30,976 | torch | 5.50 s | 669 MB | 0.05 | 265112 |
| 256 px | 65,536 | triton (fused) | 1.93 s | 1066 MB | 0.29 | 38533 |
| 256 px | 65,536 | torch | 11.61 s | 1247 MB | 0.05 | 231448 |
| 320 px | 102,400 | triton (fused) | 3.02 s | 1606 MB | 0.29 | 36390 |
| 320 px | 102,400 | torch | 18.16 s | 1870 MB | 0.05 | 218898 |

*`scan GSU/s` = scan state-updates (rows x tokens, summed over every scan call) / forward time; `--` when a model has no scan.*

*`auto` is the tier a first forward actually resolves to, so it can differ from the fused tier when the probe falls back; the row says which tier ran. Rows are measured `auto` first with the allocator emptied between rows, so each row's figures belong to that row.*
<!-- bench:end -->

Measured end-to-end through ComfyUI's native nodes (RTX 3060 Laptop 6 GB, pure-PyTorch scan,
64×64 input — one tile, so the tile policy above does not apply):

| Checkpoint | Output | Native nodes | `MambaIR Upscale Image` |
|---|---|---|---|
| `mambairv2_lightSR_x2.pth` | 128×128 | 0.9 s | 0.4 s |
| `mambairv2_classicSR_Large_x2/x3/x4` | up to 256×256 | 3.3–3.7 s | 1.9 s |
| `MambaIR-real.pth` (v1 x4) | 256×256 | 5.8 s | 4.3 s |
| `MambaIR_CAR_q10.pth` (v1, 1-ch, ×1) | 64×64 | 3.2 s | 2.7 s |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Checkpoint missing from the dropdown | not in `ComfyUI/models/upscale_models/`; press **R** to refresh |
| "model is not supported" | not a MambaIR file, or a family not implemented yet (see the table above) |
| Much slower than the table | the Triton backend is not in use — check the `MambaIR: scan backend = …` startup line |
| Blurry result for a denoiser | a scale-1 model does not upscale; use it before/after an upscaler |
| Out of memory | lower `tile` on the node, or close other models; the pack retries at half the cap by itself |
| `tile` seems ignored | values below 128 are clamped to 128 (the OOM floor) and a warning is logged; the widget steps in 128s so it should not be selectable |

Console diagnostics are prefixed `MambaIR:` (registered families, scan backend, missing-weights hint).

## Development

Source style is documented in [`docs/style.md`](docs/style.md) — Google docstrings, comments that
state invariants rather than history, and measurements living in the README or a spec instead of in
source. `tests/test_docstrings.py` enforces it without any extra dependency; `python -m ruff check .`
enforces the same rules in an editor (install the `dev` extra).

`mambair/archs/mambair*.py` are vendored from upstream and must stay byte-identical so
`PROVENANCE.md` diffs keep working — they are exempt from the style rules by path, and
`tests/test_docs_only.py` checks their hashes.

```bash
python tests/run_tests.py                     # unit + integration, no ComfyUI needed
RUN_GOLDEN=1 python tests/run_tests.py        # the seven-checkpoint golden sweeps
python tests/docs_baseline.py check           # docstring-stripped AST drift
python tests/bench.py --write                 # regenerate the benchmark table above
```

## Licence and what it obliges you to

**AGPL-3.0-or-later** for this pack — see `LICENSE`.

Two licences live here on purpose. `mambair/archs/` is derived from the MambaIR project, belongs to
the MambaIR authors, and stays **Apache-2.0** (`LICENSE-APACHE-2.0`; provenance, upstream commit,
the file-by-file diff summary and the parity test that guards it are in `PROVENANCE.md`).
Everything else — the scan, inference, tiling, the node, the tests — is AGPL-3.0-or-later, and each
file says so in its header.

**What §13 means for you.** Deploying this pack behind a network service (ComfyUI's own HTTP server,
a hosted workflow, an API) obliges you to offer everyone who interacts with it the Corresponding
Source of your version, including your changes. Running it locally for yourself triggers nothing.
ComfyUI is GPL-3.0, and its §13 expressly permits combining with AGPL-3.0 code — the AGPL network
terms then apply to the combination, which is exactly what this licence is for.
