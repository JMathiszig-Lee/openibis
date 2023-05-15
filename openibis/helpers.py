import math
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def nEpochs(eeg, Fs, stride):
    """
    This function calculates the number of epochs.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.
      Fs: The sampling frequency of the EEG data, in Hz.
      stride: The stride of the EEG data, in seconds.

    Returns:
      The number of epochs.
    """

    # Calculate the total number of samples.
    nSamples = len(eeg)

    # Calculate the number of epochs.
    nEpochs = (nSamples - Fs) // stride + 10

    n_stride = int(Fs * stride)
    N = int(np.floor((len(eeg) - Fs) / n_stride) - 10)
    return N, n_stride


def meanBandPower(
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

    if from_freq > to_freq:
        raise ValueError("The start of the frequency band must be less than or equal to the end.")

    # Calculate the indices of the frequency band.
    # bandIndices = np.arange(from * bins, to * bins, bins)

    # Calculate the mean power in the frequency band.
    meanPower = np.mean(psd[
        :,
        bandRange(from_freq, to_freq, bins)[0] : bandRange(from_freq, to_freq, bins)[1],
    ])
    return meanPower


def bandRange(from_freq: float, to_freq: float, bins: float) -> np.ndarray:
    """
    Return indices of bins for a given frequency band.

    Args:
        from_freq (float): Start frequency of the band.
        to_freq (float): End frequency of the band.
        bins (float): Bin size.

    Returns:
        np.ndarray: Indices of bins for the given frequency band.
    """
    # return np.arange(int(from_freq / bins), int(to_freq / bins)).reshape(-1, 1)
    # return np.arange((from_freq / bins)-1, to_freq / bins).astype(int)
    start = int(from_freq / bins) - 1
    end = int(to_freq / bins)
    return (start, end)


def baseline(x):
    """
    This function fits a baseline to the input data.

    Args:
      x: A 1D NumPy array of data.

    Returns:
      The baseline.
    """
    #bard
    # Calculate the number of samples.
    nSamples = len(x)

    # Fit a linear polynomial to the first and last half of the data.
    p = np.polyfit(np.arange(nSamples // 2), x[: nSamples // 2], 1)
     # Return the baseline.
    return p[0] * np.arange(nSamples) + p[1]

    #gpt4
    # v = np.linspace(0, 1, len(x))
    # return v * np.linalg.lstsq(v[:, np.newaxis], x, rcond=None)[0]
   


def bound(x, lowerBound, upperBound):
    """
    This function fixes the input between bounds.

    Args:
      x: The input data.
      lowerBound: The lower bound.
      upperBound: The upper bound.

    Returns:
      The fixed data.
    """

    # Check that the bounds are valid.
    if lowerBound > upperBound:
        raise ValueError(
            "The lower bound must be less than or equal to the upper bound."
        )

    # Fix the data between bounds.
    return np.clip(x, lowerBound, upperBound)


# @njit
def segment(eeg, start, number, stride):
    """
    This function extracts a segment of the EEG data.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.
      start: The start index of the segment.
      number: The number of samples in the segment.
      stride: The stride of the segment.

    Returns:
      The segment.
    """

    # Check that the segment is valid.
    if start + number * stride > len(eeg):
        raise ValueError("The segment is out of bounds.")

    # Extract the segment.
    start = int(start * stride)
    num_samples = int(number * stride)
    seg = eeg[start : start + num_samples]

    # Return the segment.
    return seg


def isNotBurstSuppressed(BSRmap, n, p):
    """
    This function checks if an EEG epoch is not burst-suppressed.

    Args:
      BSRmap: A boolean array indicating whether each epoch is burst-suppressed.
      n: The index of the epoch.
      p: The probability threshold.

    Returns:
      True if the epoch is not burst-suppressed, False otherwise.
    """

    # Check if the epoch is burst-suppressed.
    if BSRmap[n]:
        return False

    # Check if the epoch is above the probability threshold.
    return np.mean(BSRmap[n - p : n + p]) < p


def timeRange(seconds, n, stride):
    """
    This function returns the indices of the most recent time points.

        Args:
      seconds: The number of seconds to consider.
      n: The index of the current epoch.
      stride: The stride of the epochs.

    Returns:
      A NumPy array of indices.
    """

    # Check that the number of seconds is valid.
    if seconds <= 0:
        raise ValueError("The number of seconds must be greater than 0.")

    if n == 0:
        return [0]
    # Calculate the start index of the time range.
    start = n - (seconds / stride)

    # Calculate the end index of the time range.
    end = n

    # Return the indices of the time range.
    return np.arange(max(0, int(start)), end)
    # return (max(0, int(start)), end)
    

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
    """
    This function evaluates a piecewise function.

    Args:
    x: The input data.
    xp: The breakpoints.
    yp: The values of the function at the breakpoints.

    Returns:
    The value of the function at the input data.
    """

    # Check that the breakpoints are sorted.
    if not np.all(xp[:-1] <= xp[1:]):
        raise ValueError("The breakpoints must be sorted.")
    conditions = [x < xp[0], (x >= xp[0]) & (x < xp[1]), x >= xp[1]]
    functions = [lambda x: yp[0], 
                 lambda x: ((yp[1] - yp[0]) / (xp[1] - xp[0])) * (x - xp[0]) + yp[0], 
                 lambda x: yp[1]]
    return np.piecewise(x, conditions, functions)
    return y


#   # Find the index of the closest breakpoint.
#   i = np.searchsorted(xp, x)

#   # Return the value of the function at the closest breakpoint.
#   return yp[i]


def scurve(x, Eo, Emax, x50, xwidth):
    """
    This function evaluates a logistic S-curve.

    Args:
      x: The input data.
      Eo: The output at x = 0.
      Emax: The maximum output.
      x50: The x-coordinate of the 50% point.
      xwidth: The width of the S-curve.

    Returns:
      The value of the S-curve at the input data.
    """

    # Check that the parameters are valid.
    # if Eo < 0 or Emax < 0 or x50 < 0 or xwidth < 0:
    #     raise ValueError("The parameters must be non-negative.")

    # Calculate the value of the S-curve.
    y = Eo + (Emax - Eo) / (1 + np.exp((x - x50) / xwidth))

    return y
