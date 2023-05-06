import math

import numpy as np
from numpy.lib import stride_tricks
from scipy.interpolate import interp1d
from typing import Tuple


def n_epochs(eeg, Fs=128, stride=0.5):
    n_stride = int(Fs * stride)
    N = int(np.floor((len(eeg) - Fs) / n_stride) - 10)
    return N, n_stride


def mean_band_power(
    psd: np.ndarray, from_freq: float, to_freq: float, bins: float
) -> float:
    """
    Calculate mean band power in dB.

    Args:
        psd (np.ndarray): Power spectral density array.
        from_freq (float): Starting frequency of the band.
        to_freq (float): Ending frequency of the band.
        bins (float): Bin size.

    Returns:
        float: Mean band power in dB.
    """
    v = psd[:, band_range(from_freq, to_freq, bins)]
    y = np.mean(10 * np.log10(v[~np.isnan(v)]))
    return y


def band_range(from_freq: float, to_freq: float, bins: float) -> np.ndarray:
    """
    Return indices of bins for a given frequency band.

    Args:
        from_freq (float): Start frequency of the band.
        to_freq (float): End frequency of the band.
        bins (float): Bin size.

    Returns:
        np.ndarray: Indices of bins for the given frequency band.
    """
    return np.arange(int(np.ceil(from_freq / bins)), int(np.floor(to_freq / bins)) + 1)



def baseline(x):
    v = np.linspace(0, 1, len(x))
    return v * np.linalg.lstsq(v[:, np.newaxis], x, rcond=None)[0]


def bound(x: float, lower: float, upper: float) -> float:
    """
    Bound a value between lower and upper limits.

    Args:
        x (float): Value to be bounded.
        lower (float): Lower bound.
        upper (float): Upper bound.

    Returns:
        float: Bounded value.
    """
    return max(lower, min(x, upper))


def segment(eeg, start, num_segments, n_stride):
    start = int(start * n_stride)
    num_samples = int(num_segments * n_stride)
    return eeg[start : start + num_samples]


def is_not_burst_suppressed(BSRmap: np.ndarray, n: int, p: int) -> bool:
    """
    Checks if an epoch does not contain any burst suppression.

    Args:
        BSRmap (np.ndarray): Burst suppression map.
        n (int): Current epoch index.
        p (int): Number of previous epochs to check.

    Returns:
        bool: True if the epoch does not contain any burst suppression, otherwise False.
    """
    y = not ((n < p) or any(BSRmap[n + np.arange(1 - p, 1)]))
    return y


def time_range(seconds: int, n: int, stride: float) -> np.ndarray:
    """
    Get the indices for the most recent time points.

    Args:
        seconds (int): Time range in seconds.
        n (int): Current epoch index.
        stride (float): Stride between EEG windows.

    Returns:
        np.ndarray: Indices for the most recent time points.
    """
    y = np.arange(max(1, n - int(seconds / stride) + 1), n + 1)
    return y


def prctmean(x: np.ndarray, lo: float, hi: float) -> float:
    """
    Calculate the mean of values within a specified percentile range.

    Args:
        x (np.ndarray): Input array.
        lo (float): Lower percentile bound.
        hi (float): Upper percentile bound.

    Returns:
        float: Mean of values within the specified percentile range.
    """
    v = np.percentile(x, [lo, hi])
    y = np.mean(x[(x >= v[0]) & (x <= v[1])])
    return y


def piecewise(x, xp, yp):
    x_bounds = (x >= xp[0]) & (x <= xp[-1])
    y = interp1d(xp, yp, bounds_error=False)(x[x_bounds])
    return y


def s_curve(x, Eo, Emax, x50, xwidth):
    y = Eo - Emax / (1 + np.exp((x - x50) / xwidth))
    return y
