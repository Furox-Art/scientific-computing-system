"""Signal processing tools."""

from cds.signals.filters import (
    BandFilterCoefficients,
    FilterCoefficients,
    apply_filter,
    butter_bandpass,
    butter_bandstop,
    butter_highpass,
    butter_lowpass,
    moving_median,
)
from cds.signals.processing import (
    convolve,
    dft,
    fft2,
    fft_radix2,
    idft,
    ifft2,
    low_pass_filter,
    power_spectrum,
)
from cds.signals.stft import (
    STFTResult,
    frame_signal,
    stft,
    window,
)

__all__ = [
    # Fourier toolkit (processing.py)
    "dft",
    "idft",
    "fft_radix2",
    "fft2",
    "ifft2",
    "convolve",
    "power_spectrum",
    "low_pass_filter",
    # Digital filter design + application (filters.py)
    "FilterCoefficients",
    "BandFilterCoefficients",
    "butter_lowpass",
    "butter_highpass",
    "butter_bandpass",
    "butter_bandstop",
    "apply_filter",
    "moving_median",
    # Time-frequency analysis (stft.py)
    "STFTResult",
    "frame_signal",
    "stft",
    "window",
]
