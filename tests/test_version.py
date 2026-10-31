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

"""REQ-V1..2: one version number, agreed on by everyone who publishes it.

The version lives in three places by construction: the metadata a registry reads
(``pyproject.toml``), the attribute a ``import`` sees (``__init__.py``), and the
internal package module (``mambair/__init__.py``). They are hand-copied, and
nothing in a release pipeline compares them -- which is how a pack ships as
``0.9.0`` while the registry serves ``1.0.0`` and a user reports a bug against a
number that never existed.

The values are parsed, not imported: ``__init__.py`` registers architectures with
spandrel on import, and a version test has no business doing that.
"""

import ast
import os
import re
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: a plain final-release number: not a pre-release (rc/b/dev), not a local build
#: (`+abc`), not a range. Which release number is the maintainter's call -- this file
#: asserts the shape and the agreement, never that the pack is "old enough" for 1.0.
RELEASE = re.compile(r"^\d+\.\d+\.\d+$", re.ASCII)


def _declared(path):
    """The string assigned to ``__version__`` at module level in ``path``."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "__version__":
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path} assigns no module-level __version__")


def _packaged():
    """The ``[project] version`` a registry or ``pip`` would read."""
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_the_three_version_copies_agree():
    """REQ-V1: packaging metadata, the node pack and the package module say the same thing."""
    versions = {
        "pyproject.toml": _packaged(),
        "__init__.py": _declared("__init__.py"),
        "mambair/__init__.py": _declared("mambair/__init__.py"),
    }
    assert len(set(versions.values())) == 1, f"version drift: {versions}"


def test_the_version_is_a_final_release():
    """REQ-V2: a publishable number, not a pre-release or a local build.

    Deliberately says nothing about *which* major. 0.x means "the interface may still move"; 1.0.0
    means "it will not". That is a decision for the maintainer at release time, not a rule the test
    suite gets to enforce -- a test that demanded 1.x would turn a policy choice into a chore.
    """
    version = _packaged()
    assert RELEASE.match(version), f"not a final release number: {version!r}"


def test_installed_metadata_matches_the_source_when_installed():
    """REQ-V1: if this tree is installed, the installed metadata must not lag the source."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("comfyui-mambair")
    except PackageNotFoundError:
        return "skip"
    assert installed == _packaged(), (installed, _packaged())
