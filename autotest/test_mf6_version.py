"""
Tests for the MODFLOW 6 version checks.

MODFLOW 6 6.9.0 moved the LAK and SFR input arrays into the input context, and
support for earlier releases is deprecated. A stand-in API reports the version.

Cases:
  - parse_version        : release numbers and the development flag are read,
                           and an unreadable version is (0, 0, 0).
  - deprecated_version   : releases before 6.9.0 are deprecated; 6.9.0
                           development builds and later are not.
  - input_context        : a release is decided by its version, a 6.9.0
                           development build by the address, and a failed
                           get_version as an old release.
  - deprecation_warning  : a deprecated version raises a FutureWarning; a
                           supported one does not.
"""

import logging
import pathlib as pl
import sys
import types
import warnings

import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.adj import Mf6Adj
from mf6adj.utils.utils_modflow import (
    has_input_context,
    is_deprecated_mf6_version,
    parse_mf6_version,
)


class _Api:
    """Stand-in for the MODFLOW 6 API that reports a version and input names."""

    def __init__(self, version, names=()):
        self.version = version
        self.names = list(names)

    def get_version(self):
        if self.version is None:
            raise OSError("access violation")
        return self.version

    def get_input_var_names(self):
        return self.names


@pytest.mark.parametrize(
    "version, expected",
    [
        ("6.8.1", ((6, 8, 1), False)),
        ("6.9.0.dev0", ((6, 9, 0), True)),
        ("6.9.0.dev0+2ff6148.dirty", ((6, 9, 0), True)),
        ("6.10.0", ((6, 10, 0), False)),
        ("", ((0, 0, 0), False)),
    ],
)
def test_parse_version(version, expected):
    assert parse_mf6_version(version) == expected


@pytest.mark.parametrize(
    "version, expected",
    [
        ("", True),
        ("6.7.0", True),
        ("6.8.1", True),
        ("6.9.0.dev0", False),
        ("6.9.0", False),
        ("6.10.0.dev0", False),
    ],
)
def test_deprecated_version(version, expected):
    assert is_deprecated_mf6_version(version) is expected


@pytest.mark.parametrize(
    "version, names, expected",
    [
        (None, (), False),
        ("6.8.1", ("__INPUT__/SF/SFR-1/MANNING",), False),
        ("6.9.0.dev0", (), False),
        ("6.9.0.dev0", ("__INPUT__/SF/SFR-1/MANNING",), True),
        ("6.9.0", (), True),
        ("6.10.0.dev0", (), True),
    ],
)
def test_input_context(version, names, expected):
    api = _Api(version, names)
    assert has_input_context("sf", "sfr-1", "manning", api) is expected


@pytest.mark.parametrize(
    "version, deprecated",
    [(None, True), ("6.8.1", True), ("6.9.0.dev0", False)],
)
def test_deprecation_warning(version, deprecated):
    adj = types.SimpleNamespace(
        _gwf=_Api(version),
        logger=types.SimpleNamespace(logger=logging.getLogger("test_mf6_version")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Mf6Adj._get_gwf_version(adj)
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == int(deprecated)
