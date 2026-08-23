"""
Tests for drains sitting on the corner of the drain flow function.

A drain without a drainage depth switches on and off at its elevation, and a
converged solution leaves drains sitting exactly there because a drain pins the
head at its elevation. The derivative is minus the conductance from above and
zero from below, so crediting the measure with the conductance makes the adjoint
state unbounded.

Cases:
  - test_corner_mask            : only active entries within the tolerance of
                                  their elevation are selected, and a zero
                                  tolerance disables the check.
  - test_drop_corner_entries    : a corner entry is zeroed, a discharging entry
                                  is kept, and the original array is unchanged.
  - test_group_without_elevation: a package with no elevation, such as general
                                  head, is left alone.
  - test_corner_is_scale_free   : a drain exactly on its elevation is dropped
                                  for any tolerance.
"""

import pathlib as pl
import sys

import numpy as np
import pytest

try:
    from mf6adj.boundary import drain_corner_entries, drop_corner_entries
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    from mf6adj.boundary import drain_corner_entries, drop_corner_entries


class FakeGroup(dict):
    """Stand-in for a forward-solution package group."""

    def __contains__(self, key):
        return dict.__contains__(self, key)


def _group(nodelist, hcof, elev):
    return FakeGroup(
        nodelist=np.array(nodelist, dtype=int),
        hcof=np.array(hcof, dtype=float),
        elev=np.array(elev, dtype=float),
    )


def test_corner_mask():
    """Only active entries within the tolerance of their elevation are selected."""
    # one drain on the corner, one discharging, one already inactive
    group = _group(
        nodelist=[1, 2, 3], hcof=[-100.0, -100.0, 0.0], elev=[10.0, 10.0, 10.0]
    )
    head = np.array([10.0 + 1.0e-12, 12.0, 5.0])

    mask = drain_corner_entries(group, head, tol=1.0e-6)
    assert mask.tolist() == [True, False, False]

    # a zero criterion disables the check
    mask = drain_corner_entries(group, head, tol=0.0)
    assert not mask.any()


def test_drop_corner_entries():
    """The corner entry is zeroed and the discharging entry is untouched."""
    group = _group(nodelist=[1, 2], hcof=[-100.0, -250.0], elev=[10.0, 10.0])
    head = np.array([10.0 + 1.0e-12, 12.0])

    hcof, ncorner = drop_corner_entries(group, group["hcof"], head, 1.0e-6)
    assert ncorner == 1
    assert hcof.tolist() == [0.0, -250.0]

    # the original array is not modified in place
    assert group["hcof"].tolist() == [-100.0, -250.0]

    hcof, ncorner = drop_corner_entries(group, group["hcof"], head, 0.0)
    assert ncorner == 0


def test_group_without_elevation():
    """A package with no elevation, such as general head, is left alone."""
    group = FakeGroup(
        nodelist=np.array([1, 2]),
        hcof=np.array([-100.0, -250.0]),
        bhead=np.array([10.0, 10.0]),
    )
    head = np.array([10.0, 10.0])
    hcof, ncorner = drop_corner_entries(group, group["hcof"], head, 1.0e-6)
    assert ncorner == 0
    assert hcof.tolist() == [-100.0, -250.0]


@pytest.mark.parametrize("tol", [1.0e-9, 1.0e-6, 1.0e-4])
def test_corner_is_scale_free(tol):
    """A drain exactly on its elevation is dropped for any criterion."""
    group = _group(nodelist=[1], hcof=[-3339.77], elev=[336.59])
    head = np.array([336.59])
    mask = drain_corner_entries(group, head, tol)
    assert mask.tolist() == [True]
