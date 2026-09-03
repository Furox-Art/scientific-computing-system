# Pandas Interoperability Tutorial

`cds.data_analysis` ships an **optional** bridge that converts between
CDS's pure-Python [`DataSet`][cds.data_analysis.DataSet] (a list of row
dicts) and a pandas `DataFrame`. The core package stays zero-dependency;
install the extra and the conversion functions become available:

```bash
pip install "scientific-computing-system[pandas]"
# or, to pull every optional dependency at once:
pip install "scientific-computing-system[all]"
```

The two functions live in `cds.data_analysis` and are lazily resolved on
first access, importing the package without pandas still works.

```python
from cds.data_analysis import DataSet, to_dataframe, from_dataframe
```

## 1. Why a bridge?

`DataSet` is deliberately tiny: no external libraries, trivially
serialisable, and easy to reason about in educational code. But real
datasets often arrive as a `DataFrame`. The bridge lets you:

- load / clean in pandas, then analyse with CDS helpers, **or**
- build a `DataSet` programmatically and hand it to pandas for plotting
  or I/O.

Conversions are lossless for the scalar types `DataSet` supports
(`int | float | str | bool | None`). `None` cells map to pandas `NaN`
and back, matching pandas convention.

## 2. From `DataSet` to `DataFrame`

Each row dict becomes a DataFrame row; column order follows
`ds.columns`.

```python
ds = DataSet(
    [
        {"product": "widget", "price": 9.99, "units": 100},
        {"product": "gadget", "price": 14.50, "units": 50},
        {"product": "gizmo", "price": 3.25, "units": 200},
    ]
)

df = to_dataframe(ds)
print(df)
#   product  price  units
# 0  widget   9.99    100
# 1  gadget  14.50     50
# 2   gizmo   3.25    200

print(list(df.columns))  # ['product', 'price', 'units']
```

Column order is preserved exactly as defined in the source rows, handy
when downstream pandas code expects a specific layout.

## 3. From `DataFrame` to `DataSet`

Going the other way, each DataFrame row becomes a dict keyed by column
name. pandas `NaN` is normalised to `None` so the result honours the
`DataSet` scalar contract.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "city": ["Istanbul", "Ankara", "Izmir"],
        "temp": [28.5, 27.0, float("nan")],
    }
)

ds = from_dataframe(df)
print(ds.columns)  # ['city', 'temp']
print(ds.column("city"))  # ['Istanbul', 'Ankara', 'Izmir']
print(ds.column("temp"))  # [28.5, 27.0, None]  (NaN -> None)
```

## 4. Round-trip fidelity

`to_dataframe` → `from_dataframe` is an identity for the supported
scalar types, so you can cross the boundary freely:

```python
original = DataSet(
    [
        {"id": 1, "temp": 36.5, "flag": True},
        {"id": 2, "temp": None, "flag": False},
    ]
)

restored = from_dataframe(to_dataframe(original))
assert restored.columns == original.columns
assert restored.column("id") == [1, 2]
assert restored.column("flag") == [True, False]
```

## 5. A realistic workflow

Mix pandas I/O with CDS analysis: read a CSV with pandas, convert to a
`DataSet` for `group_by` aggregation, then push the result back to
pandas for plotting.

```python
import pandas as pd
from cds.data_analysis import from_dataframe, to_dataframe

# Load with pandas (robust CSV parsing, type inference).
raw = pd.read_csv("sales.csv")

# Move into CDS for declarative grouping + aggregation.
ds = from_dataframe(raw)
means = ds.group_by("region").mean("revenue")
print(means)  # {'EMEA': 1234.5, 'APAC': 980.0, ...}

# Hand aggregated results back to pandas for a chart.
summary = pd.DataFrame({"region": list(means), "avg_revenue": list(means.values())})
summary.plot.bar(x="region", y="avg_revenue")
```

## 6. Empty data

Both directions handle the empty case gracefully:

```python
empty = DataSet([])
df = to_dataframe(empty)
print(len(df))  # 0

back = from_dataframe(pd.DataFrame())
print(len(back))  # 0
```

## 7. When pandas is missing

If the extra was never installed, calling either function raises a
clear `ImportError` with install guidance, rather than a cryptic stack
trace:

```python
from cds.data_analysis import to_dataframe

to_dataframe(ds)
# ImportError: pandas is an optional dependency. Install it with
# `pip install scientific-computing-system[pandas]` ...
```

The package itself imports fine without pandas; only the bridge
functions require it.

---

That is the whole bridge: two functions, lossless round-trips, and a
clean escape hatch when pandas is absent. Use it to slot CDS into an
existing pandas pipeline without giving up either tool.
