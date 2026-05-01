"""Tests for the parquet caching decorator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tapt.data.cache import _hash_args, cached_parquet, clear_cache


class TestHashArgs:
    """The argument hash must be deterministic across calls and stable across types."""

    def test_same_args_produce_same_hash(self):
        h1 = _hash_args(("AAPL",), {"start": "2024-01-01"})
        h2 = _hash_args(("AAPL",), {"start": "2024-01-01"})
        assert h1 == h2

    def test_different_args_produce_different_hashes(self):
        h1 = _hash_args(("AAPL",), {"start": "2024-01-01"})
        h2 = _hash_args(("MSFT",), {"start": "2024-01-01"})
        assert h1 != h2

    def test_kwarg_order_does_not_matter(self):
        h1 = _hash_args((), {"a": 1, "b": 2})
        h2 = _hash_args((), {"b": 2, "a": 1})
        assert h1 == h2

    def test_timestamps_normalize_correctly(self):
        h1 = _hash_args((), {"start": pd.Timestamp("2024-01-01")})
        h2 = _hash_args((), {"start": pd.Timestamp("2024-01-01 00:00:00")})
        assert h1 == h2


class TestCachedParquetDecorator:
    """The decorator should cache results and avoid re-running expensive functions."""

    def test_first_call_computes_second_call_reads_cache(self, tmp_cache_dir: Path):
        call_count = {"n": 0}

        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def fake_loader(seed: int) -> pd.DataFrame:
            call_count["n"] += 1
            return pd.DataFrame({"x": [seed, seed + 1]})

        result1 = fake_loader(seed=42)
        result2 = fake_loader(seed=42)
        assert call_count["n"] == 1
        pd.testing.assert_frame_equal(result1, result2)

    def test_different_args_trigger_separate_cache_entries(self, tmp_cache_dir: Path):
        call_count = {"n": 0}

        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def fake_loader(seed: int) -> pd.DataFrame:
            call_count["n"] += 1
            return pd.DataFrame({"x": [seed]})

        fake_loader(seed=1)
        fake_loader(seed=2)
        fake_loader(seed=1)  # cache hit
        assert call_count["n"] == 2

    def test_use_cache_false_bypasses_cache(self, tmp_cache_dir: Path):
        call_count = {"n": 0}

        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def fake_loader(seed: int) -> pd.DataFrame:
            call_count["n"] += 1
            return pd.DataFrame({"x": [seed]})

        fake_loader(seed=1)
        fake_loader(seed=1, use_cache=False)
        assert call_count["n"] == 2

    def test_refresh_cache_forces_recomputation(self, tmp_cache_dir: Path):
        call_count = {"n": 0}

        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def fake_loader(seed: int) -> pd.DataFrame:
            call_count["n"] += 1
            return pd.DataFrame({"x": [seed + call_count["n"]]})

        r1 = fake_loader(seed=1)
        r2 = fake_loader(seed=1, refresh_cache=True)
        assert call_count["n"] == 2
        assert r1.iloc[0, 0] != r2.iloc[0, 0]

    def test_series_round_trips_correctly(self, tmp_cache_dir: Path):
        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def fake_loader() -> pd.Series:
            return pd.Series([1, 2, 3], name="rates")

        r1 = fake_loader()
        r2 = fake_loader()
        assert isinstance(r1, pd.Series)
        assert isinstance(r2, pd.Series)
        pd.testing.assert_series_equal(r1, r2)

    def test_non_dataframe_return_raises_type_error(self, tmp_cache_dir: Path):
        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def bad_loader() -> dict:
            return {"not": "a dataframe"}

        with pytest.raises(TypeError, match="DataFrame or Series"):
            bad_loader()

    def test_namespaces_are_isolated(self, tmp_cache_dir: Path):
        @cached_parquet("ns_a", cache_dir=tmp_cache_dir)
        def loader_a() -> pd.DataFrame:
            return pd.DataFrame({"x": [1]})

        @cached_parquet("ns_b", cache_dir=tmp_cache_dir)
        def loader_b() -> pd.DataFrame:
            return pd.DataFrame({"x": [2]})

        loader_a()
        loader_b()
        assert (tmp_cache_dir / "ns_a").exists()
        assert (tmp_cache_dir / "ns_b").exists()


class TestClearCache:
    """Cache clearing should remove files but leave directory structure intact."""

    def test_clear_removes_cached_files(self, tmp_cache_dir: Path):
        @cached_parquet("test_namespace", cache_dir=tmp_cache_dir)
        def fake_loader(seed: int) -> pd.DataFrame:
            return pd.DataFrame({"x": [seed]})

        fake_loader(seed=1)
        fake_loader(seed=2)
        n_removed = clear_cache(namespace="test_namespace", cache_dir=tmp_cache_dir)
        assert n_removed >= 2  # parquet files plus meta files

    def test_clear_nonexistent_namespace_returns_zero(self, tmp_cache_dir: Path):
        n = clear_cache(namespace="does_not_exist", cache_dir=tmp_cache_dir)
        assert n == 0
