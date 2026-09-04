"""
Tests for packages the adjoint forms no terms for.

The package types the adjoint handles are a fixed set. A model may carry
others, and their exchange with the aquifer is in the flow matrix, so the head
sensitivities account for it. The packages themselves have none, and a measure
of one has no answer.

Cases:
  - measure_unsupported : a measure on such a package is refused while the
                          adjoint file is read, rather than failing later.
  - measure_supported   : a measure on a package that is handled is accepted.
  - measure_absent      : a name that is in no package at all is still refused
                          as missing rather than as unsupported.
"""

import pathlib as pl
import sys

import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.utils.utils_modflow import (
    SUPPORTED_PACKAGE_TYPES,
    UNSUPPORTED_STRESS_TYPES,
)
from mf6adj.utils.utils_pm_read import validate_pm_type

# a model carrying a handled package and an unhandled one
PACKAGE_DICT = {
    "dis6": ["dis"],
    "npf6": ["npf"],
    "ghb6": ["ghb-1"],
    "uzf6": ["uzf-1"],
    "csub6": ["csub-1"],
}


@pytest.mark.parametrize("pm_type", ["uzf-1", "csub-1"])
def test_measure_unsupported(pm_type):
    """A measure on a package the adjoint does not form terms for is refused."""
    with pytest.raises(Exception, match="does not form terms for"):
        validate_pm_type(pm_type, PACKAGE_DICT)


@pytest.mark.parametrize("pm_type", ["head", "ghb-1"])
def test_measure_supported(pm_type):
    """A measure the adjoint can answer is accepted."""
    validate_pm_type(pm_type, PACKAGE_DICT)


def test_measure_absent():
    """A name in no package at all is refused as missing, not as unsupported."""
    with pytest.raises(Exception, match="was not found"):
        validate_pm_type("nowhere-1", PACKAGE_DICT)


def test_supported_and_unsupported_are_distinct():
    """No package type is both formed and not formed."""
    assert not set(SUPPORTED_PACKAGE_TYPES) & set(UNSUPPORTED_STRESS_TYPES)
