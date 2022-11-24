import numpy as np
from helpers import find_baseline, moving_average, number_of_epochs, isNotBurstSuppressed
from scipy import signal
from scipy.fft import fft

async def suppression(eeg, stride, Fs: int = 128):
    """_summary_

    Args:
        eeg (_type_): _description_
        stride (_type_): _description_
        Fs (int, optional): Sampling frequency. Defaults to 128Hz.

    Returns:
        BSRmap: Is the segment of eeg supressed
        BSR: the percentage of burst suppression in the previous 63 seconds
    """
    (
        nEpochs,
        nStride,
    ) = number_of_epochs(eeg, stride)
    bsrMap = []

    # for each epoch determine if it is supressed
    for i in range(nEpochs):
        start = i * nStride
        stop = (i + 1) * nStride
        segment = eeg[start:stop]
        base = find_baseline(segment)
        diff = all(x <= 5 for x in abs(segment - base))
        bsrMap.append(diff)

    window = (63 / stride) - 1
    bsr = 100 * moving_average(bsrMap, window)

    return bsrMap, bsr
 

async def log_power_ratios(eeg, stride, bsrmap, Fs: int = 128):
    nEpochs, nStride = number_of_epochs(eeg, stride)
    b, a = signal.butter(2, (0.65 / (Fs / 2)), "high")
    filtered_eeg = signal.lfilter(b, a, eeg)
    for n in nEpochs:
        if isNotBurstSuppressed(bsrmap, n, 4):
            pass



async def power_spectral_density(signal):
    f = fft(np.blackman(len(signal)) * (signal-find_baseline(signal)))
    return np.abs(f) ** 2
