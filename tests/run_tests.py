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
"""Dependency-free test runner (pytest is not installed in ComfyUI's env).

Usage:  python tests/run_tests.py [-k pattern] [--list]

Test modules are plain ``test_*.py`` files with ``test_*`` functions and bare
``assert``s, so they also run under pytest if it is available.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def discover(pattern: str | None):
    for name in sorted(os.listdir(HERE)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        if pattern and pattern not in name:
            continue
        path = os.path.join(HERE, name)
        spec = importlib.util.spec_from_file_location(name[:-3], path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name[:-3]] = module
        spec.loader.exec_module(module)
        yield module


def main(argv):
    pattern = None
    if "-k" in argv:
        pattern = argv[argv.index("-k") + 1]
    if "--list" in argv:
        for module in discover(pattern):
            for name in dir(module):
                if name.startswith("test_"):
                    print(f"{module.__name__}::{name}")
        return 0

    passed, failed, skipped = 0, 0, 0
    failures = []
    t0 = time.time()
    for module in discover(pattern):
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            fn = getattr(module, name)
            if not callable(fn):
                continue
            label = f"{module.__name__}::{name}"
            started = time.time()
            try:
                outcome = fn()
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ == "Skipped":
                    skipped += 1
                    print(f"SKIP {label} ({exc})")
                    continue
                failed += 1
                print(f"FAIL {label} ({time.time() - started:.2f}s)")
                failures.append((label, traceback.format_exc()))
                continue
            if outcome == "skip":
                skipped += 1
                print(f"SKIP {label}")
                continue
            passed += 1
            print(f"ok   {label} ({time.time() - started:.2f}s)")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped in {time.time() - t0:.1f}s")
    for label, tb in failures:
        print(f"\n--- {label}\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
