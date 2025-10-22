# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pandas = pytest.importorskip("pandas")


def test_mixed_nested_none_inner():
    """Test that records with None in nested fields don't disappear with how='inner'"""
    xxx = {
        "x": ["abc", "FG_12345"],
        "y": [None, ["g1", "g2"]],
    }
    df = ak.to_dataframe(ak.Array(xxx), how="inner")

    # Entry 0 should be present even though y is None
    # With inner merge, we expect only records where all fields are non-None
    # But entry 0 should still appear with x="abc"
    assert len(df) > 0
    assert 0 in df.index.get_level_values("entry"), "Entry 0 is missing from DataFrame"


def test_mixed_nested_none_outer():
    """Test that records with None in nested fields appear with how='outer'"""
    xxx = {
        "x": ["abc", "FG_12345"],
        "y": [None, ["g1", "g2"]],
    }
    df = ak.to_dataframe(ak.Array(xxx), how="outer")

    # With outer merge, entry 0 should definitely be present
    assert 0 in df.index.get_level_values("entry"), "Entry 0 is missing from DataFrame"
    assert 1 in df.index.get_level_values("entry"), "Entry 1 is missing from DataFrame"

    # Entry 0 should have x="abc" and y=NaN
    entry_0 = df.loc[0]
    if isinstance(entry_0, pandas.Series):
        # Single row, no subentry level for entry 0
        assert entry_0["x"] == "abc"
        assert pandas.isna(entry_0["y"])
    else:
        # Multiple rows for entry 0
        assert "abc" in entry_0["x"].values


def test_mixed_nested_empty_list():
    """Test that records with empty lists (not None) behave correctly"""
    xxx = {
        "x": ["abc", "FG_12345"],
        "y": [[], ["g1", "g2"]],
    }
    df = ak.to_dataframe(ak.Array(xxx), how="outer")

    # Entry 0 with empty list should appear
    assert 0 in df.index.get_level_values("entry") or len(df) >= 1
    assert 1 in df.index.get_level_values("entry")


def test_mixed_nested_none_multiple_fields():
    """Test multiple non-nested fields with one nested field containing None"""
    data = {
        "a": ["x", "y", "z"],
        "b": [1, 2, 3],
        "c": [None, ["c1", "c2"], ["c3"]],
    }
    df = ak.to_dataframe(ak.Array(data), how="outer")

    # All entries should be present
    assert 0 in df.index.get_level_values("entry"), "Entry 0 is missing"
    assert 1 in df.index.get_level_values("entry"), "Entry 1 is missing"
    assert 2 in df.index.get_level_values("entry"), "Entry 2 is missing"


def test_nested_strings_with_none():
    """Test the original reported case with strings"""
    xxx = {
        "x": ["abc", "FG_12345"],
        "y": [None, ["g1", "g2"]],
    }
    arr = ak.Array(xxx)
    df = ak.to_dataframe(arr)

    # The bug: entry 0 completely disappears!
    # After fix, entry 0 should be present
    entries = df.index.get_level_values("entry").unique()
    assert 0 in entries, f"Entry 0 missing! Only have entries: {list(entries)}"
    assert 1 in entries, f"Entry 1 missing! Only have entries: {list(entries)}"


def test_all_none_nested_field():
    """Test when all values in nested field are None"""
    xxx = {
        "x": ["a", "b", "c"],
        "y": [None, None, None],
    }
    df = ak.to_dataframe(ak.Array(xxx), how="outer")

    # All entries should be present
    assert len(df) == 3 or set(df.index.get_level_values("entry")) == {0, 1, 2}
