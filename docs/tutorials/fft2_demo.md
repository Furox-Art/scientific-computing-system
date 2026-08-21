# 2-D FFT Tutorial

`cds.signals` extends its from-scratch Fourier toolkit to two dimensions with
`fft2` / `ifft2` (row-column Cooley-Tukey) — pure Python, no NumPy.

## 1. Forward transform of a small image

A 4×4 "image" is just a list of rows. `fft2` returns the complex spectrum of
the same shape.

```python
from cds.signals import fft2

image = [
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
    [4.0, 5.0, 6.0, 7.0],
]

spectrum = fft2(image)
```

The DC component `F[0][0]` equals the sum of all pixels:

```
DC component (sum of all pixels):
  F[0][0] = 64.0  (expected 64.0)
```

The magnitude spectrum shows where the signal's energy lives. For a smooth
ramp like this, almost everything is concentrated at low frequencies:

```
Magnitude spectrum |F[u][v]|:
   64.00   11.31    8.00   11.31
   11.31    0.00    0.00    0.00
    8.00    0.00    0.00    0.00
   11.31    0.00    0.00    0.00
```

## 2. Inverse transform (round-trip)

`ifft2` inverts the transform. The reconstruction error against the original
image is machine-precision:

```python
from cds.signals import ifft2

recovered = ifft2(spectrum)
max_err = max(abs(recovered[i][j].real - image[i][j]) for i in range(4) for j in range(4))
print(f"Max reconstruction error: {max_err:.2e}")
#   Max reconstruction error: 0.00e+00
```

## When to use it

2-D FFTs power classic image-processing tasks: filtering (low/high-pass in
the frequency domain), convolution (via the convolution theorem), and
compression (energy compaction). CDS keeps the implementation readable so you
can follow exactly how rows then columns are transformed.

Run the full demo with:

```bash
python examples/fft2_demo.py
```
