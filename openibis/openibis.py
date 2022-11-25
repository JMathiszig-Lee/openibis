import numpy as np
from .helpers import find_baseline, moving_average, number_of_epochs, isNotBurstSuppressed, get_segment
from scipy import signal
from scipy.fft import fft
from tqdm import tqdm

async def depth_of_anaesthesia(eeg):
    bsrmap, bsr = await suppression(eeg, 0.5)
    components = await log_power_ratios(eeg, 0.5, bsrmap)


async def suppression(eeg, stride=0.5, Fs: int = 128):
    """_summary_

    Args:
        eeg (_type_): _description_
        stride (_type_): _description_
        Fs (int, optional): Sampling frequency. Defaults to 128Hz.

    Returns:
        BSRmap: Is the segment of eeg supressed
        BSR: the percentage of burst suppression in the previous 63 seconds
    """
    nEpochs, nStride = number_of_epochs(eeg, stride, Fs)
    bsrMap = []

    # for each epoch determine if it is supressed
    print('Checking epochs for burst suppression')
    for i in tqdm(range(nEpochs)):
        segment = get_segment(eeg, i, 2, nStride)
        try:
            base = find_baseline(segment)
        except:
            base = 0
        diff = all(x <= 5 for x in abs(segment - base))

        bsrMap.append(diff)

    window = (63 / stride) - 1
    bsr = 100 * moving_average(bsrMap, int(window))

    return bsrMap, bsr
 

async def log_power_ratios(eeg, stride, bsrmap, Fs: int = 128):
    nEpochs, nStride = number_of_epochs(eeg, stride)
    b, a = signal.butter(2, (0.65 / (Fs / 2)), "high")

    #setup array for power spectral densities
    # psd = np.empty([nEpochs, int(4*nStride/2)])
    psd = np.empty([nEpochs, 256])
    psd[:] = np.nan
    print(psd.shape)

    filtered_eeg = signal.lfilter(b, a, eeg)
    print('generating power spectral densities')
    for i in tqdm(range(nEpochs)):
        if isNotBurstSuppressed(bsrmap, i, 4):
            psd[i] = await power_spectral_density(get_segment(filtered_eeg, i+4, 4, nStride))
            # print(i, psd)
            #TODO add in sawtooth detector

    print(psd)

async def power_spectral_density(signal):
    # try: 
    #     f = fft(np.blackman(len(signal)) * (signal-find_baseline(signal)))
    # except:
    #     f = fft(np.blackman(len(signal)) * (signal))
    f = fft(np.blackman(len(signal)) * signal)
    return np.abs(f) ** 2


