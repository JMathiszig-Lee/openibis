import math
import peakutils

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

