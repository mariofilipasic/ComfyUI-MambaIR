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

"""MambaIR for ComfyUI -- internal package.

Public surface used by the node pack itself:

* :mod:`mambair.scan`           -- selective scan backends
* :mod:`mambair.infer`          -- state dict -> family + hyperparameters
* :mod:`mambair.spandrel_arch`  -- architectures registered with spandrel
* :mod:`mambair.tiling`         -- bounded, feathered inference
* :mod:`mambair.apply`          -- descriptor + ComfyUI IMAGE -> IMAGE
"""

__version__ = "0.1.0"
