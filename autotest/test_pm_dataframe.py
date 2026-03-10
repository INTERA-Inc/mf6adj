"""Unit tests for PerfMeas DataFrame instantiation and entry validation."""

import pathlib as pl
import sys

import numpy as np
import pandas as pd
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj import PerfMeas, PerfMeasRecord


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_record(**kwargs):
    """Return a PerfMeasRecord with sensible defaults, overridden by kwargs."""
    defaults = dict(
        kper=0, kstp=0, inode=10, pm_type="head",
        pm_form="direct", weight=1.0, obsval=0.0,
    )
    defaults.update(kwargs)
    return PerfMeasRecord(**defaults)


def _make_df(**overrides):
    """Return a single-row DataFrame with required columns, overridden by overrides."""
    row = dict(
        kper=0, kstp=0, inode=10, pm_type="head",
        pm_form="direct", weight=1.0, obsval=0.0,
    )
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# PerfMeas.__init__ – mutual exclusion of pm_entries / df
# ---------------------------------------------------------------------------

class TestInitMutualExclusion:
    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Either 'pm_entries' or 'df'"):
            PerfMeas("pm1")

    def test_both_raises(self):
        entry = _make_record()
        df = _make_df()
        with pytest.raises(ValueError, match="not both"):
            PerfMeas("pm1", pm_entries=[entry], df=df)

    def test_pm_entries_only_ok(self):
        pm = PerfMeas("pm1", pm_entries=[_make_record()])
        assert pm.name == "pm1"

    def test_df_only_ok(self):
        pm = PerfMeas("pm1", df=_make_df())
        assert pm.name == "pm1"


# ---------------------------------------------------------------------------
# PerfMeas._entries_from_dataframe
# ---------------------------------------------------------------------------

class TestEntriesFromDataframe:
    def test_required_columns_missing_raises(self):
        df = _make_df().drop(columns=["weight", "pm_form"])
        with pytest.raises(ValueError, match="missing required column"):
            PerfMeas._entries_from_dataframe(df)

    def test_single_missing_column_named_in_error(self):
        df = _make_df().drop(columns=["obsval"])
        with pytest.raises(ValueError, match="obsval"):
            PerfMeas._entries_from_dataframe(df)

    def test_returns_correct_types(self):
        entries = PerfMeas._entries_from_dataframe(_make_df())
        assert len(entries) == 1
        assert isinstance(entries[0], PerfMeasRecord)

    def test_required_fields_mapped_correctly(self):
        df = _make_df(kper=2, kstp=3, inode=99, pm_type="ghb6",
                      pm_form="direct", weight=2.5, obsval=-1.0)
        entry = PerfMeas._entries_from_dataframe(df)[0]
        assert entry._kper == 2
        assert entry._kstp == 3
        assert entry.inode == 99
        assert entry.pm_type == "ghb6"
        assert entry.pm_form == "direct"
        assert entry.weight == 2.5
        assert entry.obsval == -1.0

    def test_optional_kij_absent(self):
        """k, i, j columns absent → PerfMeasRecord attrs are None."""
        entry = PerfMeas._entries_from_dataframe(_make_df())[0]
        assert entry._k is None
        assert entry._i is None
        assert entry._j is None

    def test_optional_kij_present(self):
        df = _make_df(k=0, i=3, j=7)
        entry = PerfMeas._entries_from_dataframe(df)[0]
        assert entry._k == 0
        assert entry._i == 3
        assert entry._j == 7

    def test_multiple_rows(self):
        rows = [
            dict(kper=0, kstp=0, inode=i, pm_type="head",
                 pm_form="direct", weight=1.0, obsval=0.0)
            for i in range(5)
        ]
        entries = PerfMeas._entries_from_dataframe(pd.DataFrame(rows))
        assert len(entries) == 5
        assert [e.inode for e in entries] == list(range(5))

    def test_numpy_int_inode(self):
        """inode supplied as numpy int64 should be handled without error."""
        df = _make_df(inode=np.int64(42))
        entry = PerfMeas._entries_from_dataframe(df)[0]
        assert entry.inode == 42


# ---------------------------------------------------------------------------
# PerfMeas._validate_entries
# ---------------------------------------------------------------------------

class TestValidateEntries:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no entries found"):
            PerfMeas._validate_entries("pm1", [])

    def test_mixed_pm_forms_raises(self):
        entries = [
            _make_record(pm_form="direct"),
            _make_record(pm_form="residual"),
        ]
        with pytest.raises(ValueError, match="mixed 'pm_forms'"):
            PerfMeas._validate_entries("pm1", entries)

    def test_flux_type_residual_form_raises(self):
        entries = [_make_record(pm_type="ghb6", pm_form="residual")]
        with pytest.raises(ValueError, match="residual"):
            PerfMeas._validate_entries("pm1", entries)

    def test_head_residual_ok(self):
        """head pm_type with residual pm_form is valid."""
        entries = [_make_record(pm_type="head", pm_form="residual")]
        PerfMeas._validate_entries("pm1", entries)  # should not raise

    def test_flux_direct_ok(self):
        """Non-head pm_type with direct pm_form is valid."""
        entries = [_make_record(pm_type="ghb6", pm_form="direct")]
        PerfMeas._validate_entries("pm1", entries)  # should not raise

    def test_valid_multiple_entries_ok(self):
        entries = [_make_record(inode=i) for i in range(3)]
        PerfMeas._validate_entries("pm1", entries)  # should not raise


# ---------------------------------------------------------------------------
# End-to-end: PerfMeas built from df matches one built from pm_entries
# ---------------------------------------------------------------------------

class TestDataframeMatchesPmEntries:
    def _make_entries(self):
        return [
            PerfMeasRecord(0, 0, 5,  "head", "direct", 1.0, 0.0),
            PerfMeasRecord(0, 0, 10, "head", "direct", 2.0, 0.0),
        ]

    def _make_df(self):
        return pd.DataFrame([
            dict(kper=0, kstp=0, inode=5,  pm_type="head", pm_form="direct", weight=1.0, obsval=0.0),
            dict(kper=0, kstp=0, inode=10, pm_type="head", pm_form="direct", weight=2.0, obsval=0.0),
        ])

    def test_same_number_of_entries(self):
        pm_e = PerfMeas("pm1", pm_entries=self._make_entries())
        pm_d = PerfMeas("pm1", df=self._make_df())
        assert len(pm_e._entries) == len(pm_d._entries)

    def test_entry_attributes_match(self):
        pm_e = PerfMeas("pm1", pm_entries=self._make_entries())
        pm_d = PerfMeas("pm1", df=self._make_df())
        for e, d in zip(pm_e._entries, pm_d._entries):
            assert e.inode == d.inode
            assert e.weight == d.weight
            assert e.pm_type == d.pm_type
            assert e.pm_form == d.pm_form
            assert e._kper == d._kper
            assert e._kstp == d._kstp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
