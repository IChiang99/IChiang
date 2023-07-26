# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
Enumeration of supported optimiser algorithms.

Note that it may not be the case that all backends support all
algorithms. Each `Optimiser` implementation must state which of these
algorithms it supports.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


class _AlgorithmMeta(enum.EnumMeta):
    def __getitem__(self, s: str) -> Algorithm:
        try:
            return super().__getitem__(s)
        except KeyError:
            if s == "DIRECT-L":
                # special case for backward compatibility
                return super().__getitem__("DIRECT_L")
            raise ValueError(f"No such Algorithm value '{s}'.")


class Algorithm(enum.Enum, metaclass=_AlgorithmMeta):
    """Enumeration of available optimisation algorithms."""

    SLSQP = enum.auto()
    COBYLA = enum.auto()
    SBPLX = enum.auto()
    MMA = enum.auto()
    BFGS = enum.auto()
    DIRECT = enum.auto()
    DIRECT_L = enum.auto()
    CRS = enum.auto()
    ISRES = enum.auto()


@dataclass
class AlgorithmTolerance:
    """Algorithm tolerances container"""

    ftol_abs: Optional[float] = None
    ftol_rel: Optional[float] = None
    xtol_abs: Optional[float] = None
    xtol_rel: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary without Nones"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AlgorithmDefaultTolerances:
    """Default Algorithm tolerances"""

    SLSQP: AlgorithmTolerance = field(
        default_factory=lambda: AlgorithmTolerance(ftol_abs=1e-6)
    )
    COBYLA: AlgorithmTolerance = field(
        default_factory=lambda: AlgorithmTolerance(ftol_abs=1e-4, xtol_abs=1e-4)
    )
    SBPLX: AlgorithmTolerance = field(default_factory=AlgorithmTolerance)
    MMA: AlgorithmTolerance = field(default_factory=AlgorithmTolerance)
    BFGS: AlgorithmTolerance = field(
        default_factory=lambda: AlgorithmTolerance(xtol_rel=0)
    )
    DIRECT: AlgorithmTolerance = field(
        default_factory=lambda: AlgorithmTolerance(ftol_rel=1e-4)
    )
    DIRECT_L: AlgorithmTolerance = field(default_factory=AlgorithmTolerance)
    CRS: AlgorithmTolerance = field(default_factory=AlgorithmTolerance)
    ISRES: AlgorithmTolerance = field(default_factory=AlgorithmTolerance)
