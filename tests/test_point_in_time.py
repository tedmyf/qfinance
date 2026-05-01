"""Tests for point-in-time discipline.

These tests are paranoid by design: the cost of false positives is small,
the cost of silent leakage is a backtest result you cannot trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tapt.data.point_in_time import PointInTimeFrame, enforce_as_of


class TestPointInTimeFrameSingleIndex:
    """Single-DatetimeIndex frames should filter correctly on as_of."""

    def test_basic_truncation(self, sample_prices: pd.DataFrame):
        pit = PointInTimeFrame(sample_prices)
        result = pit.as_of(sample_prices.index[10])
        assert len(result) == 11  # 0 through 10 inclusive
        assert result.index[-1] == sample_prices.index[10]

    def test_string_date_works(self, sample_prices: pd.DataFrame):
        pit = PointInTimeFrame(sample_prices)
        date_str = sample_prices.index[5].strftime("%Y-%m-%d")
        result = pit.as_of(date_str)
        assert result.index[-1] <= pd.Timestamp(date_str)

    def test_date_before_data_returns_empty(self, sample_prices: pd.DataFrame):
        pit = PointInTimeFrame(sample_prices)
        result = pit.as_of("2000-01-01")
        assert len(result) == 0

    def test_date_after_data_returns_all(self, sample_prices: pd.DataFrame):
        pit = PointInTimeFrame(sample_prices)
        result = pit.as_of("2099-01-01")
        pd.testing.assert_frame_equal(result, sample_prices)

    def test_inclusive_boundary(self, sample_prices: pd.DataFrame):
        """The as-of date itself should be included in the result."""
        pit = PointInTimeFrame(sample_prices)
        boundary = sample_prices.index[5]
        result = pit.as_of(boundary)
        assert boundary in result.index


class TestPointInTimeFrameMultiIndex:
    """MultiIndexed frames should filter on the date level."""

    def test_auto_detects_date_level(self, multi_index_panel: pd.DataFrame):
        pit = PointInTimeFrame(multi_index_panel)
        result = pit.as_of("2024-01-05")
        max_date = result.index.get_level_values("date").max()
        assert max_date <= pd.Timestamp("2024-01-05")

    def test_explicit_date_level(self, multi_index_panel: pd.DataFrame):
        pit = PointInTimeFrame(multi_index_panel, date_level="date")
        result = pit.as_of("2024-01-03")
        unique_dates = result.index.get_level_values("date").unique()
        assert all(d <= pd.Timestamp("2024-01-03") for d in unique_dates)

    def test_invalid_date_level_raises(self, multi_index_panel: pd.DataFrame):
        with pytest.raises(ValueError, match="not in index names"):
            PointInTimeFrame(multi_index_panel, date_level="nonexistent")


class TestPointInTimeFrameValidation:
    """Construction should reject frames that lack a usable date index."""

    def test_non_dataframe_input_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            PointInTimeFrame([1, 2, 3])  # type: ignore

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            PointInTimeFrame(df)

    def test_multiindex_without_date_level_raises(self):
        idx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="No DatetimeIndex level"):
            PointInTimeFrame(df)


class TestEnforceAsOfDecorator:
    """The decorator should defensively truncate returned DataFrames."""

    def test_decorator_truncates_when_inner_function_forgets(self):
        @enforce_as_of()
        def lazy_loader(as_of: str | None = None) -> pd.DataFrame:
            # Deliberately ignores as_of to simulate a bug
            return pd.DataFrame(
                {"x": [1, 2, 3, 4, 5]},
                index=pd.date_range("2024-01-01", periods=5),
            )

        result = lazy_loader(as_of="2024-01-03")
        assert len(result) == 3
        assert result.index.max() == pd.Timestamp("2024-01-03")

    def test_decorator_passes_through_when_as_of_is_none(self):
        @enforce_as_of()
        def loader(as_of: str | None = None) -> pd.DataFrame:
            return pd.DataFrame(
                {"x": [1, 2, 3]},
                index=pd.date_range("2024-01-01", periods=3),
            )

        result = loader()
        assert len(result) == 3

    def test_decorator_handles_multiindex_returns(self):
        @enforce_as_of()
        def loader(as_of: str | None = None) -> pd.DataFrame:
            idx = pd.MultiIndex.from_product(
                [pd.date_range("2024-01-01", periods=5), ["A", "B"]],
                names=["date", "ticker"],
            )
            return pd.DataFrame({"v": np.arange(10)}, index=idx)

        result = loader(as_of="2024-01-03")
        max_date = result.index.get_level_values("date").max()
        assert max_date <= pd.Timestamp("2024-01-03")

    def test_decorator_passes_through_non_dataframe_returns(self):
        @enforce_as_of()
        def loader(as_of: str | None = None) -> dict:
            return {"not": "a dataframe"}

        result = loader(as_of="2024-01-01")
        assert result == {"not": "a dataframe"}

    def test_custom_param_name(self):
        @enforce_as_of(date_param="cutoff")
        def loader(cutoff: str | None = None) -> pd.DataFrame:
            return pd.DataFrame(
                {"x": [1, 2, 3, 4]},
                index=pd.date_range("2024-01-01", periods=4),
            )

        result = loader(cutoff="2024-01-02")
        assert len(result) == 2
