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


"""REQ-1.x / REQ-2.x / REQ-6.x: the docstring gate.

Dependency-free on purpose (NFR-2): this is the enforcement that runs in the normal test suite.
`pyproject.toml` carries the equivalent `ruff` configuration for editors that have it, but nothing
here needs ruff installed.

What it checks:
* the twelve Tier-1 symbols carry Google sections;
* every other public module/class-level callable has a docstring unless it is waived, and every
  waiver refers to a symbol that still exists (a stale waiver is an omission hiding in plain sight);
* summary-line and section-order rules;
* every `Hyperparams` field is described.
"""

import ast
import contextlib
import importlib
import inspect
import os
import sys

import docstring_waivers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

TIER1 = (
    ("mambair.scan", "selective_scan_fn"),
    ("mambair.scan", "selective_scan_ref"),
    ("mambair.scan", "_scan_torch"),
    ("mambair.scan_triton", "selective_scan_triton"),
    ("mambair.infer", "Hyperparams"),
    ("mambair.infer", "infer"),
    ("mambair.infer", "detect_family"),
    ("mambair.tiling", "tiled_forward"),
    ("mambair.tiling", "bounded_forward"),
    ("mambair.apply", "apply_descriptor"),
    ("mambair.spandrel_arch", "ModelRunner.run"),
    ("mambair.node_core", "run_node"),
)

SECTIONS = ("Args:", "Returns:", "Yields:", "Raises:", "Attributes:", "Note:")


def _module_name(relative):
    if relative.endswith("__init__.py"):
        relative = relative[: -len("/__init__.py")] if "/" in relative else ""
    else:
        relative = relative[: -len(".py")]
    return relative.replace("/", ".")


def _qualify(relative, node, parents=()):
    """Dotted path relative to the pack root, e.g. ``nodes:MambaIRUpscaleImageV1.execute``."""
    module = _module_name(relative)
    parts = [p for p in (*parents, node.name) if p]
    dotted = ".".join(parts)
    return f"{module}:{dotted}" if module else f"{dotted}"


def _definitions(relative):
    """Yield (qualname, docstring or None, lineno) for module- and class-level defs."""
    path = os.path.join(ROOT, relative)
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    def walk(nodes, parents=()):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield _qualify(relative, node, parents), ast.get_docstring(node), node.lineno
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, (*parents, node.name))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                yield from walk(node.body, parents)
                for extra in ("orelse", "finalbody", "handlers"):
                    for sub in getattr(node, extra, []) or []:
                        yield from walk(getattr(sub, "body", [sub]), parents)

    yield from walk(tree.body)


def _docstring(module_path, qualname):
    parts = qualname.split(".")
    obj = importlib.import_module(module_path)
    for part in parts:
        obj = getattr(obj, part)
    return inspect.getdoc(obj) or ""


def test_tier1_symbols_have_google_sections():
    problems = []
    for module, name in TIER1:
        doc = _docstring(module, name)
        if not doc:
            problems.append(f"{module}.{name}: no docstring")
            continue
        # a dataclass documents its fields instead of its arguments
        required = ("Attributes:",) if name == "Hyperparams" else ("Args:", "Returns:")
        problems += [f"{module}.{name}: no {section}" for section in required if section not in doc]
    assert not problems, "\n".join(problems)


def test_tier1_raises_are_documented():
    """REQ-2.3: every reachable raise is named in a Raises: section."""
    problems = []
    for module, name in TIER1:
        obj = importlib.import_module(module)
        for part in name.split("."):
            obj = getattr(obj, part)
        tree = ast.parse(inspect.getsource(obj).strip())

        def own_raises(node):
            """Raises in this body, ignoring nested defs (a class does not raise via methods)."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(child, ast.Raise):
                    yield child
                yield from own_raises(child)

        has_raise = any(own_raises(tree))
        if has_raise and "Raises:" not in (inspect.getdoc(obj) or ""):
            problems.append(f"{module}.{name}: raises but has no Raises: section")
    assert not problems, "\n".join(problems)


def test_public_definitions_are_documented_or_waived():
    missing = []
    seen = set()
    for relative in MODULES:
        for qualname, doc, lineno in _definitions(relative):
            key = qualname
            if key in docstring_waivers.WAIVERS:
                seen.add(key)
                continue
            if not doc:
                missing.append(f"{key} (line {lineno})")
    assert not missing, "public definitions without a docstring (add one or waive it in " \
                        f"tests/docstring_waivers.py):\n  " + "\n  ".join(missing)
    stale = sorted(set(docstring_waivers.WAIVERS) - seen)
    assert not stale, f"waivers that no longer match a symbol: {stale}"


def test_summary_lines_and_section_order():
    problems = []
    for relative in MODULES:
        for qualname, doc, lineno in _definitions(relative):
            if not doc:
                continue
            first = doc.strip().splitlines()[0].strip()
            if len(first) > 100:
                problems.append(f"{qualname}:{lineno} summary is {len(first)} chars")
            if not first.endswith("."):
                problems.append(f"{qualname}:{lineno} summary does not end with a period")
            order = [doc.index(s) for s in SECTIONS if s in doc]
            if order != sorted(order):
                problems.append(f"{qualname}:{lineno} sections are out of order")
    assert not problems, "\n".join(problems)


def test_the_style_guide_shows_a_real_docstring():
    """REQ-2.1: `docs/style.md`'s worked example is a copy of a live function, so it must match.

    The example had drifted from `selective_scan_fn` in five places while the document still claimed
    "this function is real". A style guide that teaches the previous shape of a docstring is worse
    than one with no example, so the copy is pinned to the source it advertises.
    """
    from mambair.scan import selective_scan_fn

    with open(os.path.join(ROOT, "docs", "style.md"), encoding="utf-8") as fh:
        blocks = fh.read().split("```python")[1:]
    example = next(b.split("```", 1)[0] for b in blocks if "def selective_scan_fn" in b)
    node = next(n for n in ast.parse(example).body if isinstance(n, ast.FunctionDef))

    assert ast.get_docstring(node) == inspect.getdoc(selective_scan_fn), (
        "docs/style.md's worked example no longer matches mambair.scan.selective_scan_fn"
    )
    assert [a.arg for a in node.args.args] == list(
        inspect.signature(selective_scan_fn).parameters
    ), "the example's signature drifted from the real one"


def test_hyperparams_fields_are_documented():
    from mambair.infer import Hyperparams

    doc = Hyperparams.__doc__ or ""
    missing = [name for name in Hyperparams.__dataclass_fields__ if f"{name}:" not in doc]
    assert not missing, f"undocumented Hyperparams fields: {missing}"
    assert "Attributes:" in doc
