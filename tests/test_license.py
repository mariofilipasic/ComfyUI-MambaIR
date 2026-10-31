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

"""REQ-L1..4: the project is AGPL-3.0-or-later, and the vendored code is not.

Two licences live here on purpose. ``mambair/archs/`` is derived from the MambaIR
project (Apache-2.0) and stays Apache-2.0; everything else is this pack's own work.
A file header that gets copied into the wrong side is both a licence misstatement
and a provenance break, so the split is asserted rather than trusted.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: upstream-derived code: Apache-2.0, never re-headered (see PROVENANCE.md)
DERIVED_DIR = os.path.join("mambair", "archs")

#: our own licence, in the machine-readable form every source file carries
SPDX = "SPDX-License-Identifier: AGPL-3.0-or-later"

#: the copyright line, stated once. Every project file must carry exactly this string; the metadata
#: in pyproject.toml must name the same holder. Re-headering the tree by hand is how a project ends
#: up with three spellings of its owner and an unclear provenance story.
HOLDER = "Mario Filipasic"
COPYRIGHT = f"Copyright (C) 2026 {HOLDER}"

#: the whole block, asserted as a unit. Checking only two of its lines is how a header loses its
#: warranty notice or its licence URL to a line-ending rewrite and still passes.
HEADER = (
    "# MambaIR for ComfyUI\n"
    "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    f"# {COPYRIGHT}\n"
    "#\n"
    "# This program is free software: you can redistribute it and/or modify it under the terms of the\n"
    "# GNU Affero General Public License as published by the Free Software Foundation, either version 3\n"
    "# of the License, or (at your option) any later version. It is distributed in the hope that it will\n"
    "# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or\n"
    "# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.\n"
    "# You should have received a copy of it with this program; if not, see\n"
    "# <https://www.gnu.org/licenses/>.\n"
    "\n"
)


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _python_files():
    """Every ``.py`` file in the repo, split into (project, derived)."""
    project, derived = [], []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.relpath(os.path.join(base, name), ROOT)
            if path.startswith(".specs") or path.startswith(".pi"):
                continue
            (derived if path.startswith(DERIVED_DIR) else project).append(path)
    return sorted(project), sorted(derived)


def test_the_licence_file_is_the_agpl():
    """REQ-L1: a verbatim AGPL-3.0, not a summary and not a modified copy."""
    text = _read("LICENSE")
    head = text.splitlines()[:3]
    assert head[0].strip() == "GNU AFFERO GENERAL PUBLIC LICENSE", head
    assert "Version 3, 19 November 2007" in head[1], head
    assert "Copyright (C) 2007 Free Software Foundation, Inc." in text
    assert "changing it is not allowed" in text
    # §13 is the network clause -- the reason this licence was chosen over GPL-3.0
    assert "13. Remote Network Interaction" in text
    assert text.rstrip().endswith("<https://www.gnu.org/licenses/>.")


def test_the_apache_text_is_still_shipped():
    """REQ-L2: the vendored files' licence must remain available to their recipients."""
    text = _read("LICENSE-APACHE-2.0")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text


def test_project_files_carry_the_agpl_header():
    """REQ-L3: every file this pack owns states its licence, in its first lines."""
    project, _ = _python_files()
    assert len(project) > 30, f"discovered only {len(project)} files -- is the walk broken?"
    missing = []
    for path in project:
        head = _read(path).replace("\r\n", "\n")
        if SPDX not in head[:200]:
            missing.append(f"{path} (no SPDX identifier near the top)")
        elif not head.startswith(HEADER):
            missing.append(f"{path} (header differs from the canonical block)")
    assert not missing, "project files without the AGPL header: " + ", ".join(missing)


def test_derived_files_keep_their_own_licence():
    """REQ-L4: upstream code is Apache-2.0 -- no AGPL header, and its notice stays."""
    _, derived = _python_files()
    assert derived, "the vendored architectures disappeared from the tree"
    wrong = []
    for path in derived:
        text = _read(path)
        if "SPDX-License-Identifier: AGPL" in text:
            wrong.append(f"{path}: AGPL header on upstream-derived code")
        if "Apache" not in text:
            wrong.append(f"{path}: Apache provenance notice missing")
    assert not wrong, wrong


def test_packaging_metadata_agrees_with_the_licence_file():
    """REQ-L1: the metadata a registry reads and the text a recipient gets say the same thing."""
    pyproject = _read("pyproject.toml")
    assert 'license = { text = "AGPL-3.0-or-later" }' in pyproject, pyproject
    stale = [line for line in pyproject.splitlines()
             if line.startswith("license =") and "AGPL" not in line]
    assert not stale, f"pyproject advertises another licence: {stale}"
    assert 'license-files = ["LICENSE", "LICENSE-APACHE-2.0"]' in pyproject, (
        "both licence texts must ship in the distribution"
    )
    assert f'name = "{HOLDER}"' in pyproject, (
        f"pyproject.toml must name the same copyright holder as the headers ({HOLDER!r})"
    )
