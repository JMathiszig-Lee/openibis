from typing import Tuple

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from alive_progress import alive_bar
from scipy.signal import blackman, butter, convolve, filtfilt
from scipy.stats import percentileofscore, trim_mean

from .helpers import (
    bandRange,
    baseline,
    bound,
    isNotBurstSuppressed,
    meanBandPower,
    nEpochs,
    piecewise,
    prctmean,
    scurve,
    segment,
    timeRange,
)


def openibis(eeg):
    """
    This function calculates the depth of anesthesia from an EEG signal.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.

    Returns:
      A 1D NumPy array of depth-of-anesthesia scores.
    """

    # Set the sampling frequency and stride.
    Fs = 128
    stride = 0.5

    # Calculate the burst suppression rate (BSR).
    BSRmap, BSR = suppression(eeg, Fs, stride)

    # Calculate the other components for the depth-of-anesthesia.
    components = logPowerRatios(eeg, Fs, stride, BSRmap)

    # Mix the components and BSR together to generate the depth-of-anesthesia scores.
    depthOfAnesthesia = mixer(components, BSR)

    return depthOfAnesthesia


def suppression(eeg, Fs, stride):
    """
    This function determines whether an EEG segment is burst-suppressed and calculates BSR.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.
      Fs: The sampling frequency of the EEG data, in Hz.
      stride: The stride of the EEG data, in seconds.

    Returns:
      A tuple of two NumPy arrays:
        - The first array is a boolean array indicating whether each epoch is burst-suppressed.
        - The second array is the burst suppression rate (BSR).
    """

    # Calculate the total number of epochs.
    N, n_stride = nEpochs(eeg, Fs, stride)

    # Allocate space to store a true/false map of whether epochs are burst suppressed.
    BSRmap = np.zeros(N, dtype=bool)

    # Evaluate over the available epochs.
    with alive_bar(N, title="Supression Map") as bar:
        for n in range(N):
            x = segment(eeg, n + 6.5, 2, n_stride)
            BSRmap[n] = np.all(np.abs(x - baseline(x)) <= 5)
            bar()

    # print(BSRmap)
    # Calculate the burst suppression rate (BSR).
    BSR = np.mean(BSRmap) * 100

    return BSRmap, BSR


def gpt4suppresion(eeg, Fs=128, stride=0.5):
    N, n_stride = nEpochs(eeg, Fs, stride)
    BSRmap = np.zeros(N)
    for n in range(N):
        x = segment(eeg, n + 6.5, 2, n_stride)
        BSRmap[n] = np.all(np.abs(x - baseline(x)) <= 5)
    BSR = (
        100
        * pd.Series(BSRmap).rolling(window=int((63 / stride) - 1), min_periods=1).mean()
    )
    # print(BSRmap)
    return BSRmap, BSR


def logPowerRatios(eeg, Fs, stride, BSRmap):
    """
    This function calculates a family of components, based on log-power ratios.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.
      Fs: The sampling frequency of the EEG data, in Hz.
      stride: The stride of the EEG data, in seconds.
      BSRmap: A boolean array indicating whether each epoch is burst-suppressed.

    Returns:
      A NumPy array of three components.
    """

    # Calculate the total number of epochs.
    N, n_stride = nEpochs(eeg, Fs, stride)

    # Create a second order high-pass Butterworth filter at 0.65 Hz.
    B, A = butter(2, 0.65 / (Fs / 2), "high")

    # Filter out very low frequencies from the input EEG.
    eegHiPassFiltered = filtfilt(B, A, eeg)

    # Allocate space to store power spectral densities, 0 to 63.5 Hz with 0.5 Hz bins.
    psd = np.empty((N, 128)) * np.nan

    # Define a suppression filter in the range of 0 - 6 Hz
    SuppressionFilter = piecewise(np.arange(0, 63.5, 0.5), [0, 3, 6], [0, 0.25, 1])

    # Allocate space to store output signal components, 3 components for each epoch
    components = np.zeros((N, 3), dtype=float)

    # Evaluate over the available epochs.
    with alive_bar(N, title="epoch eval") as bar:
        for n in range(N):
            n
            # print(BSRmap)
            if not BSRmap[n]:
                psd[n, :] = powerSpectralDensity(
                    segment(eegHiPassFiltered, n + 4, 4, n_stride), Fs, stride
                )
                # print(psd[n])
                # if sawtoothDetector(eeg[n * stride : (n + 1) * stride], stride):
                #     psd[n, :] = SuppressionFilter * psd[n, :]
            # %[#YH.g1nFITIH>cdPg
            # Consider data from the most recent thirty seconds.
            thirtySec = timeRange(30, n, stride)
            # print(thirtySec)
            # print(f"psd shape: {psd.shape}")
            # print(psd[thirtySec,bandRange(39.5, 46.5, 0.5)[0] : bandRange(39.5, 46.5, 0.5)[1]])
            psd2 = psd[
                thirtySec,
                bandRange(39.5, 46.5, 0.5)[0] : bandRange(39.5, 46.5, 0.5)[1],
            ]
            # print("psd2")
            # print(psd2)
            # print(psd2.shape)
            # Calculate the VhighPowerConc.
            VhighPowerConc = np.sqrt(
                np.mean(
                    psd[
                        thirtySec,
                        bandRange(39.5, 46.5, 0.5)[0] : bandRange(39.5, 46.5, 0.5)[1],
                    ]
                    * psd[
                        thirtySec,
                        bandRange(40, 47, 0.5)[0] : bandRange(40, 47, 0.5)[1],
                    ],
                    1,
                )
            )
            # print(f"v high power = {VhighPowerConc}")
            #
            # Calculate the wholePowerConc.
            wholePowerConc = np.sqrt(
                np.mean(
                    psd[
                        thirtySec,
                        bandRange(0.5, 46.5, 0.5)[0] : bandRange(0.5, 46.5, 0.5)[1],
                    ]
                    * psd[
                        thirtySec,
                        bandRange(1, 47, 0.5)[0] : bandRange(1, 47, 0.5)[1],
                    ],
                    1,
                )
            )
            # print(f"whole power = {wholePowerConc}")

            # mid_band_power = prctmean(np.nanmean(10 * np.log10(psd[thirty_sec, :][:, band_range(11, 20, 0.5)]), axis=1), 50, 100)
            # mid_band_power = prctmean(np.nanmean(10 * np.log10(psd[thirtySec, bandRange(11, 20, 0.5)]), axis=1), 50, 100)
            band_power = 10 * np.apply_along_axis(
                np.log10,
                1,
                psd[
                    thirtySec,
                    bandRange(11, 20, 0.5)[0] : bandRange(11, 20, 0.5)[1],
                ],
            )
            # print(f"band power: {band_power}")
            # mid_band_power = np.percentile(band_power, [50, 100])
            mid_band_power = prctmean(
                np.nanmean(
                    10
                    * np.log10(
                        psd[
                            thirtySec,
                            bandRange(11, 20, 0.5)[0] : bandRange(11, 20, 0.5)[1],
                        ],
                    ),
                    axis=1,
                ),
                50,
                100,
            )
            # print(f"mid power = {mid_band_power}")

            # Calculate the component 1.
            mean_band_power = (
                meanBandPower(psd[thirtySec, :], 30, 47, 0.5) - mid_band_power
            )
            # print(f"mean band power = {mean_band_power}")
            components[n, 0] = mean_band_power
            # Calculate the component 2.
            components[n, 1] = trim_mean(
                np.log10(VhighPowerConc / wholePowerConc), 0.25
            )
            # Calculate the component 3.
            components[n, 2] = meanBandPower(
                psd[
                    thirtySec,
                    bandRange(11, 20, 0.5)[0] : bandRange(11, 20, 0.5)[1],
                ],
                11,
                20,
                5,
            )
            bar()

    return components


def powerSpectralDensity(eeg, Fs=128, stride=0.5):
    """
    This function calculates the power spectral density of the input EEG data.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.
      Fs: The sampling frequency of the EEG data, in Hz.
      stride: The stride of the EEG data, in seconds.

    Returns:
      A 2D NumPy array of power spectral densities, in units of dB.
    """
    # bard
    # # Calculate the total number of epochs.
    # N = len(eeg) // stride

    # # Create a hanning window.
    # window = np.hanning(N * stride)

    # # Calculate the power spectral densities.
    # psd = np.fft.fft(eeg * window, n=N * stride)

    # # Calculate the power spectral densities in dB.
    # psddB = 10 * np.log10(np.abs(psd))

    # # Return the power spectral densities.
    # return psddB

    # gpt4
    # def power_spectral_density(x):
    f = np.fft.fft(blackman(len(eeg)) * (eeg - baseline(eeg)))
    y = (
        2
        * np.abs(f[: len(eeg) // 2]) ** 2
        / (len(eeg) * np.sum(blackman(len(eeg)) ** 2))
    )
    return y


def sawtoothDetector(eeg, stride):
    """
    This function detects sawtooth waves in the input EEG data.

    Args:
      eeg: A 1D NumPy array of EEG data, in units of uV.
      stride: The stride of the EEG data, in seconds.

    Returns:
      A NumPy array of boolean values indicating whether each epoch contains a sawtooth wave.
    """

    # Calculate the total number of epochs.
    N = len(eeg) // stride

    # Create a sawtooth wave.
    sawtooth = np.concatenate([np.zeros(N // 2), np.ones(N // 2)])

    # Calculate the cross-correlation between the EEG data and the sawtooth wave.
    crossCorrelation = np.correlate(eeg, sawtooth, mode="full")

    # Calculate the maximum cross-correlation.
    maxCrossCorrelation = np.max(crossCorrelation)

    # Calculate the threshold for detecting sawtooth waves.
    threshold = 0.1 * maxCrossCorrelation

    # Return a boolean array indicating whether each epoch contains a sawtooth wave.
    return crossCorrelation > threshold


def mixer(components, BSR):
    """
    This function generates the output depth-of-anesthesia by converting and weighting components, BSRs.

    Args:
      components: A NumPy array of three components.
      BSR: The burst suppression rate (BSR).

    Returns:
      A NumPy array of depth-of-anesthesia scores.
    """
    print(components)
    # Map component 1 to a sedation score on a logistic S-curve.
    sedationScore = scurve(components[:, 0], 104.4, 49.4, -13.9, 5.29)
    print(sedationScore)
    # Map component 2 to a general score, linear region and S-curved region.
    generalScore = piecewise(components[:, 1], [-60.89, -30], [-40, 43.1])
    print(generalScore)
    generalScore += scurve(components[:, 1], 61.3, 72.6, -24.0, 3.55) * (
        components[:, 1] >= -30
    )

    # Convert the BSR to a BSR score using a piecewise linear function.
    bsrScore = np.piecewise(BSR, [0, 100], [50, 0])

    # Convert component 3 to a weight.
    generalWeight = piecewise(components[:, 2], [0, 5], [0.5, 1]) * (
        generalScore < sedationScore
    )

    # Convert the BSR to a weight.
    bsrWeight = piecewise(BSR, [10, 50], [0, 1])

    # Weight the sedation and general scores together.
    x = (sedationScore * (1 - generalWeight)) + (generalScore * generalWeight)

    # Compress and weight these with the BSR.
    y = (
        piecewise(x, [-40, 10, 97, 110], [0, 10, 97, 100]) * (1 - bsrWeight)
        + bsrScore * bsrWeight
    )

    return y
