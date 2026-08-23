"""Adjoint coupling for the MODFLOW 6 advanced packages.

An advanced package carries its own dependent variable - a lake stage, a reach
stage - solved in the outer iteration rather than in the solution matrix.
Holding that variable fixed gives a partial derivative, so the adjoint system
is bordered with the package's own equation. Each package keeps its terms in a
module here, leaving the adjoint solver free of package detail.
"""

from .lake import LakeCoupling, table_slope
from .lake import forward_terms as lake_forward_terms
from .sfr import SfrCoupling
from .sfr import forward_terms as sfr_forward_terms

__all__ = [
    "LakeCoupling",
    "SfrCoupling",
    "lake_forward_terms",
    "sfr_forward_terms",
    "table_slope",
]
