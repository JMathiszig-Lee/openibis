import numpy as np
from helpers import number_of_epochs, find_baseline


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
    nEpochs, nStride,  = number_of_epochs(eeg, stride)

    #for each epoch determine if it is supressed
    for i in range(nEpochs):
        start = i*nStride
        stop = (i+1)*nStride
        segment = eeg[start:stop]
        base = find_baseline(segment)
        diff = all(x<=5 for x in abs(segment-base))


    bsrMap = 0
    bsr = 0
    
    return bsrMap, bsr
