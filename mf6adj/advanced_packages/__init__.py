"""Adjoint coupling for the MODFLOW 6 advanced packages.

An advanced package carries its own dependent variable - a lake stage, a reach
stage, a well head. Holding that variable fixed gives a partial derivative, so
the package's own equation has to be part of the adjoint system. A lake or a
reach is solved in the outer iteration rather than in the solution matrix, so
the adjoint system is bordered with its equation; a multi-aquifer well is
solved with the flow equations, so its equation is already a row of the matrix.
Each package keeps its terms in a module here, leaving the adjoint solver free
of package detail.
"""

from .lake import LakeCoupling, table_slope
from .lake import forward_terms as lake_forward_terms
from .maw import MawCoupling
from .maw import forward_terms as maw_forward_terms
from .sfr import SfrCoupling
from .sfr import forward_terms as sfr_forward_terms

__all__ = [
    "LakeCoupling",
    "MawCoupling",
    "SfrCoupling",
    "lake_forward_terms",
    "maw_forward_terms",
    "sfr_forward_terms",
    "table_slope",
]
