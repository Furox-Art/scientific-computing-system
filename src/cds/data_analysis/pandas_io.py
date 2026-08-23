"""Pandas interoperability bridge for :class:`cds.data_analysis.DataSet`.

This module is **optional**: pandas is imported lazily, so the core package
remains zero-dependency. Install the extra with ``pip install ".[pandas]"``
(or ``[all]``). Importing this module without pandas installed raises a clear
``ImportError`` with install guidance.

The bridge converts between CDS's pure-Python :class:`DataSet` (a list of
row dicts) and a pandas ``DataFrame``, letting users who already live in the
pandas ecosystem move data in both directions without rewriting pipelines.

Design notes:
- Conversions are lossless for the scalar types :class:`DataSet` supports
  (``int | float | str | bool | None``).
- ``None`` cells map to ``NaN`` in pandas and back, matching pandas convention.
- Column order is preserved from the source on both directions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cds.data_analysis.dataset import DataSet, Row

if TYPE_CHECKING:
    # Imported only under TYPE_CHECKING so the module imports cleanly when
    # pandas is absent; the real import happens inside the functions.
    import pandas as pd


_PANDAS_INSTALL_HINT = (
    "pandas is an optional dependency. Install it with "
    "`pip install scientific-computing-system[pandas]` "
    "(or `pip install pandas`)."
)


def _require_pandas() -> Any:
    """Import and return the ``pandas`` module, raising a helpful error if absent."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(_PANDAS_INSTALL_HINT) from exc
    return pd


def to_dataframe(ds: DataSet) -> pd.DataFrame:
    """Convert a :class:`DataSet` to a pandas ``DataFrame``.

    Each row dict becomes a DataFrame row; column order follows
    ``ds.columns``. ``None`` values are preserved as-is (pandas renders them
    as ``NaN`` in numeric columns).

    Args:
        ds: a CDS :class:`DataSet`.

    Returns:
        A pandas ``DataFrame`` with the same rows and columns.

    Raises:
        ImportError: if pandas is not installed.
    """
    pandas_module = _require_pandas()
    rows = ds.to_list()
    columns = ds.columns
    # The DataFrame constructor is typed as returning Any in the stubs; assign
    # to an annotated local so the return type is pinned without a cast.
    df: pd.DataFrame = pandas_module.DataFrame(rows, columns=columns if columns else None)
    return df


def from_dataframe(df: pd.DataFrame) -> DataSet:
    """Convert a pandas ``DataFrame`` to a :class:`DataSet`.

    Each DataFrame row becomes a row dict keyed by column name. pandas
    ``NaN`` values are converted to ``None`` to match :class:`DataSet`'s
    scalar type contract.

    Args:
        df: a pandas ``DataFrame``.

    Returns:
        A CDS :class:`DataSet` with the same data.

    Raises:
        ImportError: if pandas is not installed.
    """
    pandas_module = _require_pandas()
    rows: list[Row] = []
    # .to_dict(orient="records") yields plain dicts; NaN -> NaN (float).
    # We normalize NaN to None to honor the DataSet scalar contract.
    raw_records = df.to_dict(orient="records")
    for record in raw_records:
        row: Row = {}
        for key, value in record.items():
            # pd.isna handles both scalar NaN and None, but raises on arrays.
            # Values here are scalars from a DataFrame cell.
            try:
                is_na = bool(pandas_module.isna(value))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                # pd.isna raises only on array-like inputs. DataSet's Row
                # contract restricts cells to scalars (int|float|str|bool|None),
                # so this branch is unreachable for conforming data; kept as a
                # guard against malformed/foreign rows slipping through.
                is_na = value is None
            row[str(key)] = None if is_na else value
        rows.append(row)
    return DataSet(rows)
