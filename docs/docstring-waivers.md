# Docstring waivers

**Policy:** a symbol may go undocumented only if the omission is **recorded with a reason**, because
an undocumented public name is a decision someone made and did not write down. The
machine-readable list is
[`tests/docstring_waivers.py`](../tests/docstring_waivers.py); `tests/test_docstrings.py` fails when a
public definition is undocumented *or* when a waiver no longer matches a real symbol, so this cannot
rot into a pile of stale exceptions.

**Scope.** The gate checks module- and class-level definitions. Local closures and functions defined
inside another function are implementation details and out of scope — which is why the four spandrel
registration classes created inside `architectures()` appear nowhere.

## Waived

| Symbol | Reason |
|---|---|
| `MambaIRUpscaleImage` and its `define_schema` / `validate_inputs` / `fingerprint_inputs` / `execute` | ComfyUI fixes these contracts; the schema *is* the documentation |
| `MambaIRExtension`, `get_node_list`, `comfy_entrypoint` | framework hooks |
| `MambaIRUpscaleImageV1` and its `INPUT_TYPES` / `IS_CHANGED` / `VALIDATE_INPUTS` / `execute` | legacy node; each method delegates to a documented `node_core` function |
| `DropPath.forward`, `DropPath.__init__` | one-line delegation / stores one attribute |
| `ModelRunner.__init__`, `__call__`, `scale`, `extra_repr` | the class docstring and `run()` cover them; `__call__` exists only to satisfy spandrel's `call_fn` |
| `_combine`, `_row_scan` | Triton kernels: `triton.jit` parses the body, so they carry `#` comments; the prefix-scan design is documented once in the module docstring |

## Not waived, deliberately

Everything in `mambair/{scan,scan_triton,infer,tiling,deterministic,apply,node_core,registry}.py`
has a docstring, including the private helpers, because each one encodes a decision that is not
recoverable from its body (why chunk-local, why a fixed probe slice, why the reference tier below
`PROBE_MIN_TOKENS`).

Vendored files under `mambair/archs/mambair*.py` are excluded by path, not by waiver: they must stay
byte-identical to upstream so `PROVENANCE.md` diffs keep working (`tests/test_docs_only.py` checks
their hashes).
