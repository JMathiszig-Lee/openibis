import math

import numpy as np
import peakutils
from numpy.lib import stride_tricks


def number_of_epochs(eeg, stride=0.5, Fs=128):
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

def moving_average(a: list, window:int):

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

def isNotBurstSuppressed(bsrmap:list[bool], n: int, strides: int) -> bool:
    """Checks for burst suppression

    Args:
        bsrmap (list[bool]): BSR map
        n (int): epoch number
        strides (int): number of strides

    Returns:
        bool: True if not burst supressed
    """
    if n < strides:
        return True
    elif any(bsrmap[n:(n+strides)]):
        #if any of the bsr map is true(there is burst suppression)
        return False
    else: 
        return True

def get_segment(eeg, start: int, number: int, nStride: int = 64):
    """_summary_

    Args:
        eeg (_type_): an eeg 
        start (_type_): start stride
        number (_type_): number of strides
        nStride (_type_): samples per stride

    Returns:
        segment: a segment of the eeg 
    """
    a = start * nStride
    b = (number*nStride) + a
    seg = eeg[int(a):int(b)]

    return seg

def saw_tooth_detector(eeg, nStride) -> bool :
    pass

def scurve(x, Eo, Emax, x50, xwidth):
    return Eo - Emax/(1+np.exp((x-x50)/xwidth))

def logistic(x, L=1, x_0=0, k=1):
    return L / (1 + np.exp(-k * (x - x_0)))