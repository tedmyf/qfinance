"""Point-in-time discipline for DataFrames.

A common source of backtest look-ahead bias is accidentally including data
that would not have been available at the time of a decision. This module
provides two tools to prevent that:

- ``PointInTimeFrame`` wraps a DataFrame and enforces a date cutoff on reads.
- ``enforce_as_of`` is a decorator that post-filters the returned DataFrame.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import pandas as pd


class PointInTimeFrame:
    """Wraps a DataFrame and provides a point-in-time filtered view.

    Parameters
    ----------
    df : pd.DataFrame
        The underlying data. Index must be a ``DatetimeIndex`` (for
        single-index frames) or a ``MultiIndex`` with at least one
        ``DatetimeIndex`` level.
    date_level : str, optional
        For MultiIndex frames, the name of the date level. Auto-detected
        if not provided.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame.
    ValueError
        If the index has no usable ``DatetimeIndex`` level, or if
        ``date_level`` is not in the index names.
    """

    def __init__(self, df: pd.DataFrame, date_level: str | None = None) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(df).__name__}")

        self._df = df

        if isinstance(df.index, pd.MultiIndex):
            if date_level is not None:
                if date_level not in df.index.names:
                    raise ValueError(
                        f"date_level={date_level!r} not in index names {list(df.index.names)}"
                    )
                self._date_level: str = date_level
            else:
                datetime_levels = [
                    name
                    for name, level in zip(df.index.names, df.index.levels)
                    if isinstance(level, pd.DatetimeIndex)
                ]
                if not datetime_levels:
                    raise ValueError(
                        "No DatetimeIndex level found in MultiIndex; "
                        "specify date_level explicitly"
                    )
                self._date_level = datetime_levels[0]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"Index must be a DatetimeIndex; got {type(df.index).__name__}"
                )
            self._date_level = ""

    def as_of(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return all rows with date index <= ``date``."""
        cutoff = pd.Timestamp(date)

        if isinstance(self._df.index, pd.MultiIndex):
            level_values = self._df.index.get_level_values(self._date_level)
            return self._df[level_values <= cutoff]

        return self._df[self._df.index <= cutoff]


def enforce_as_of(date_param: str = "as_of") -> Callable:
    """Decorator that post-filters returned DataFrames to the ``as_of`` date.

    The decorated function must accept a keyword argument named ``date_param``
    (default ``"as_of"``). When that argument is not None, the decorator
    applies ``PointInTimeFrame.as_of`` to the returned DataFrame. Non-DataFrame
    return values are passed through unchanged.

    Parameters
    ----------
    date_param : str
        Name of the keyword argument that holds the cutoff date.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            date_val = kwargs.get(date_param)
            if date_val is None or not isinstance(result, pd.DataFrame):
                return result
            return PointInTimeFrame(result).as_of(date_val)

        return wrapper

    return decorator
