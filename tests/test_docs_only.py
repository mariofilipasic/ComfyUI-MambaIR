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

"""REQ-7.1 / REQ-5.1 / REQ-8.2: the documentation pass changes prose, not code.

The baseline in `tests/docs_baseline.json` was captured from the hardened tree. Adding, rewording
or deleting a docstring cannot move it; adding a statement, changing a default, reordering
arguments or touching a vendored file will.
"""

import os

import docs_baseline

#: tokens no Python source in this pack may contain (REQ-8.2)
RENDERER_MARKUP = ("@param", "@return", ":param", ":return:", "\\param", "\\return")


def test_no_code_changed_since_the_baseline():
    if not os.path.isfile(docs_baseline.BASELINE):
        return "skip"
    changed = docs_baseline.compare(docs_baseline.load())
    assert not changed, (
        "executable structure changed since the docs baseline: "
        + "; ".join(f"{bucket}: {names}" for bucket, names in changed.items())
        + ".  If the change is intended, re-capture with "
        "`python tests/docs_baseline.py capture` in the same commit and say why."
    )


def test_vendored_architectures_are_untouched():
    """REQ-5.1: upstream files stay byte-identical so PROVENANCE diffs keep working."""
    if not os.path.isfile(docs_baseline.BASELINE):
        return "skip"
    expected = docs_baseline.load()["vendored"]
    drifted = [
        name for name, value in expected.items()
        if docs_baseline.raw_digest(name) != value
    ]
    assert not drifted, f"vendored files modified: {drifted}"


def test_no_renderer_specific_markup():
    """REQ-8.2: one docstring format serves pydoc, pdoc, editors and review."""
    offenders = []
    for name in docs_baseline.MODULES:
        with open(os.path.join(docs_baseline.ROOT, name), "r", encoding="utf-8") as fh:
            text = fh.read()
        for token in RENDERER_MARKUP:
            if token in text:
                offenders.append(f"{name}: {token!r}")
    assert not offenders, "renderer-specific markup found: " + ", ".join(offenders)
