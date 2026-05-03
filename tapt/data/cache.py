"""Parquet-backed caching decorator for data loaders.

Results are stored as Parquet files under a namespace directory, keyed by a
SHA-256 hash of the call arguments. This avoids redundant network requests
in interactive sessions and backtests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path.home() / ".tapt_cache"


def _hash_args(args: tuple, kwargs: dict) -> str:
    """Return a short deterministic hex digest for the given call arguments."""

    def _normalize(v: object) -> object:
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        return v

    payload = {
        "args": [_normalize(a) for a in args],
        "kwargs": {k: _normalize(v) for k, v in sorted(kwargs.items())},
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cached_parquet(namespace: str, cache_dir: Path | None = None) -> Callable:
    """Decorator that persists a function's DataFrame/Series result as Parquet.

    The decorated function gains two extra keyword-only parameters:
    - ``use_cache`` (bool, default True): when False, always re-run the function.
    - ``refresh_cache`` (bool, default False): re-run and overwrite the cached file.

    Parameters
    ----------
    namespace : str
        Subdirectory name under ``cache_dir`` for this function's files.
    cache_dir : Path, optional
        Root cache directory. Defaults to ``DEFAULT_CACHE_DIR`` at call time.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, use_cache: bool = True, refresh_cache: bool = False, **kwargs):
            _dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
            ns_dir = _dir / namespace
            ns_dir.mkdir(parents=True, exist_ok=True)

            key = _hash_args(args, kwargs)
            parquet_path = ns_dir / f"{key}.parquet"
            meta_path = ns_dir / f"{key}.meta.json"

            if use_cache and not refresh_cache and parquet_path.exists():
                frame = pd.read_parquet(parquet_path)
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("type") == "Series":
                        return frame.iloc[:, 0].rename(meta.get("name"))
                return frame

            result = fn(*args, **kwargs)

            if isinstance(result, pd.Series):
                meta_path.write_text(json.dumps({"type": "Series", "name": result.name}))
                result.to_frame().to_parquet(parquet_path)
            elif isinstance(result, pd.DataFrame):
                result.to_parquet(parquet_path)
            else:
                raise TypeError(
                    f"cached_parquet requires a DataFrame or Series return value; got {type(result)}"
                )

            return result

        return wrapper

    return decorator


def clear_cache(namespace: str, cache_dir: Path | None = None) -> int:
    """Remove all cached files for ``namespace``.

    Returns
    -------
    int
        Number of files removed.
    """
    _dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    ns_dir = _dir / namespace

    if not ns_dir.exists():
        return 0

    files = list(ns_dir.iterdir())
    for f in files:
        f.unlink()
    return len(files)
