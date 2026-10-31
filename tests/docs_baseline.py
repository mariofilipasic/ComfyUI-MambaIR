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

#!/usr/bin/env python
"""Docstring-stripped AST digests for the documentation pass (api-docs spec, D0.1).

    python tests/docs_baseline.py capture     # write tests/docs_baseline.json
    python tests/docs_baseline.py check       # compare, non-zero exit on drift

Why: a documentation pass must not change code, and "I only edited docstrings" is not a review
argument. Comments never appear in the AST, and docstrings are removed here entirely (present or
absent, they are ignored), so the digest is a statement about *executable structure only*. Any
statement, expression, default, annotation, decorator or ordering change moves it.

`vendored` hashes are raw bytes instead: those files must not be touched at all (REQ-5.1).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(HERE, "docs_baseline.json")

#: our own modules -- the ones the documentation pass rewrites
MODULES = (
    "__init__.py",
    "nodes.py",
    "mambair/__init__.py",
    "mambair/apply.py",
    "mambair/deterministic.py",
    "mambair/infer.py",
    "mambair/node_core.py",
    "mambair/registry.py",
    "mambair/scan.py",
    "mambair/scan_triton.py",
    "mambair/spandrel_arch.py",
    "mambair/tiling.py",
    "mambair/archs/__init__.py",
)

#: upstream code: byte-identical, no docstring edits allowed (REQ-5.1)
VENDORED = (
    "mambair/archs/mambair.py",
    "mambair/archs/mambairunet.py",
    "mambair/archs/mambairv2.py",
    "mambair/archs/mambairv2light.py",
)


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove every docstring statement, so adding or rewording one is invisible here."""
    nodes = list(ast.walk(tree))
    for node in nodes:
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del body[0]
            if not body:  # a docstring-only body must stay a valid module/class/function
                body.append(ast.Pass())
    return tree


def digest(relative: str) -> str:
    path = os.path.join(ROOT, relative)
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    return hashlib.sha256(ast.dump(strip_docstrings(ast.parse(source))).encode()).hexdigest()


def raw_digest(relative: str) -> str:
    with open(os.path.join(ROOT, relative), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def snapshot() -> dict:
    return {
        "modules": {name: digest(name) for name in MODULES},
        "vendored": {name: raw_digest(name) for name in VENDORED},
    }


def capture(path: str = BASELINE) -> dict:
    data = snapshot()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(data['modules'])} module digests and {len(data['vendored'])} vendored hashes")
    return data


def load(path: str = BASELINE) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def compare(expected: dict, actual: dict | None = None) -> dict:
    """Return {bucket: [changed module, ...]}, empty when nothing moved."""
    actual = actual or snapshot()
    changed = {}
    for bucket in ("modules", "vendored"):
        drift = sorted(
            name for name, value in expected.get(bucket, {}).items()
            if actual[bucket].get(name) != value
        )
        if drift:
            changed[bucket] = drift
    return changed


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "check"
    if command == "capture":
        capture()
        return 0
    if command == "check":
        if not os.path.isfile(BASELINE):
            print(f"no baseline at {BASELINE}; run: python tests/docs_baseline.py capture")
            return 2
        changed = compare(load())
        if not changed:
            print("no code changes since the baseline")
            return 0
        for bucket, names in changed.items():
            print(f"{bucket} changed: {', '.join(names)}")
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
