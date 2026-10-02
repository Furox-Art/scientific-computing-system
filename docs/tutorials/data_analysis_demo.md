# Data Analysis Tutorial

`cds.data_analysis` loads small CSVs, normalises and smooths series, and renders
ASCII plots.

Two containers, two shapes:

- **`DataTable`** — tabular data backed by string rows. `load_csv` returns one.
  Read cells with `.column(name)` (strings), `.column_as_float(name)` (floats),
  or `.describe()` for per-column statistics.
- **`DataSet`** — a list of `dict` rows, i.e. records rather than a grid. Its
  constructor takes `data=`, not `name=` / `observations=`.

Neither container exposes `.rows` or `.name`; use `.n_rows` / `.n_cols` and
`.shape` / `.columns`.

## 1. Load a CSV

`load_csv` takes a filesystem path, so the file has to exist first. This example
creates a small CSV and reads it back.

```python
import pathlib
import tempfile

from cds.data_analysis import load_csv

path = pathlib.Path(tempfile.gettempdir()) / "cds_demo_data.csv"
path.write_text("day,temp\n1,18.0\n2,19.5\n3,22.0\n", encoding="utf-8")

table = load_csv(str(path))
print(table.headers)  # ['day', 'temp']
print(table.n_rows)  # 3
print(table.column("temp"))
print(table.column_as_float("temp"))
```

## 2. Normalise & Smooth

These take and return plain float lists. `moving_average`'s second argument is
positional (`window`), not a keyword.

```python
from cds.data_analysis import normalize, z_score, moving_average

temps = [18.0, 19.5, 22.0, 21.5, 23.0]
print([round(v, 4) for v in normalize(temps)])  # min-max to [0, 1]
print([round(v, 4) for v in z_score(temps)])  # zero mean, unit variance
print([round(v, 4) for v in moving_average(temps, 2)])
```

## 3. ASCII Plots

Both plotting helpers **return** the rendered text, so print it to see it.

`plot_bar` takes a **dict** of label → value, not a list.

```python
from cds.data_analysis import plot_line, plot_bar

print(plot_line(temps, title="temperature"))
print(plot_bar({"mon": 100, "tue": 110, "wed": 140}, title="sales"))
```

## 4. DataSet for record-shaped data

```python
from cds.data_analysis import DataSet

ds = DataSet(
    [
        {"source": "SH0ES", "h0": 73.04},
        {"source": "Planck", "h0": 67.36},
    ]
)
print(ds.shape)  # (2, 2)
print(ds.columns)  # ['source', 'h0']
print(ds.column("h0"))  # [73.04, 67.36]
```

Run the full demo, which writes its own CSV fixture first, with:

```bash
python examples/data_analysis_demo.py
```