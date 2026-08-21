# Data Analysis Tutorial

`cds.data_analysis` loads small CSVs, normalises and smooths series, and renders ASCII plots.

## 1. Load a CSV

```python
from cds.data_analysis import load_csv

table = load_csv("data.csv")
print(table.columns, len(table.rows))
```

## 2. Normalise & Smooth

```python
from cds.data_analysis import normalize, z_score, moving_average

temps = [18.0, 19.5, 22.0, 21.5, 23.0]
print(normalize(temps))  # min-max to [0,1]
print(z_score(temps))  # zero mean, unit variance
print(moving_average(temps, window=2))
```

## 3. ASCII Plots

```python
from cds.data_analysis import plot_line, plot_bar

plot_line(temps, title="temperature")
plot_bar([100, 110, 140, 135, 160], title="sales")
```

## 4. DataSet helper

```python
from cds.data_analysis import DataSet

ds = DataSet(name="week", observations=temps)
print(ds.name, len(ds.observations))
```

Run the full demo with `python examples/data_analysis_demo.py`.
