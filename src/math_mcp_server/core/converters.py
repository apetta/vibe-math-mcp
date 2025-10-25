"""Data type conversion utilities for Polars and Pandas interoperability."""

from typing import List, Optional
import polars as pl
import pandas as pd
import numpy as np


def list_to_polars(data: List[List[float]], columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Convert nested list to Polars DataFrame.

    Args:
        data: 2D list of values
        columns: Optional column names

    Returns:
        Polars DataFrame
    """
    if columns:
        return pl.DataFrame(data, schema=columns, orient="row")
    return pl.DataFrame(data, orient="row")


def polars_to_list(df: pl.DataFrame) -> List[List[float]]:
    """Convert Polars DataFrame to nested list.

    Args:
        df: Polars DataFrame

    Returns:
        2D list of values
    """
    return df.to_numpy().tolist()


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars to Pandas (fallback only).

    Args:
        df: Polars DataFrame

    Returns:
        Pandas DataFrame
    """
    return df.to_pandas()


def list_to_numpy(data: List[List[float]]) -> np.ndarray:
    """Convert nested list to NumPy array.

    Args:
        data: 2D list of values

    Returns:
        NumPy ndarray
    """
    return np.array(data, dtype=float)


def numpy_to_list(arr: np.ndarray) -> List[List[float]]:
    """Convert NumPy array to nested list.

    Args:
        arr: NumPy array

    Returns:
        2D list of values
    """
    if arr.ndim == 1:
        return arr.tolist()
    return arr.tolist()
