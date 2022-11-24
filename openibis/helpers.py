import math

import numpy as np
import peakutils
from numpy.lib import stride_tricks


def number_of_epochs(eeg, stride, Fs=128):
    """_summary_

    Args:
        eeg (_type_): eeg vector
        stride (_type_): _description_
        Fs (int, optional): Sampling frequency. Defaults to 128.

    Returns:
        nStride: the number of samples per stride
        nEpochs: the number of epochs in the eeg given
    """

    nStride = Fs * stride
    nEpochs = math.floor((len(eeg) - Fs) / nStride) - 10

    return nEpochs, nStride


def find_baseline(segment):

    baseline_values = peakutils.baseline(segment)

    return baseline_values

def moving_average(a: list, window: int):

    return np.average(stride_tricks.sliding_window_view(a, window))

def piecewise(x, xp:list, yp:list):
    """Piecewise filter

    Args:
        x (_type_): array to have piecewise filter
        xp (_type_): conditions
        yp (_type_): functions
    """

    cl = []
    for i in range(len(xp)-1):
        cl.append(np.logical_and(x >= xp[i], x <= xp[i+1]))

    return np.piecewise(x, cl, yp)

def isNotBurstSuppressed(bsrmap:list[bool], n, strides: int) -> bool:
    """Checks for burst suppression

    Args:
        bsrmap (list): BSR map
        n (_type_): epoch number
        strides (int): number of strides

    Returns:
        bool: True if not burst supressed
    """
    if n < strides:
        return True
    elif any(bsrmap):
        #if any of the bsr map is true(there is burst suppression)
        return False
    else: 
        return True

def segment():
    pass
