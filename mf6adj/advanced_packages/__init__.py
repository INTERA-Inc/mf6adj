"""Adjoint coupling for the MODFLOW 6 advanced packages.

An advanced package carries its own dependent variable - a lake stage, a reach
stage - solved in the outer iteration rather than in the solution matrix.
Holding that variable fixed gives a partial derivative, so the adjoint system
is bordered with the package's own equation. Each package keeps its terms in a
module here, leaving the adjoint solver free of package detail.
"""

from .lake import LakeCoupling, forward_terms, table_slope

__all__ = [
    "LakeCoupling",
    "forward_terms",
    "table_slope",
]
