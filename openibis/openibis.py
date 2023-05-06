from .helpers import (
    n_epochs,
    mean_band_power,
    band_range,
    baseline,
    bound,
    segment,
    is_not_burst_suppressed,
    time_range,
    prctmean,
    piecewise,
    s_curve,
)
import numpy as np
import scipy.signal as sp_signal
import pandas as pd
from scipy.signal import butter, filtfilt, blackman, convolve
from scipy.stats import trim_mean, percentileofscore
from typing import Tuple


def openibis(eeg, Fs=128, stride=0.5):
    """
    Calculates the depth of anesthesia for the given EEG data.

    Args:
        eeg (np.ndarray): The input EEG data as a column vector (in uV).

    Returns:
        np.ndarray: Depth of anesthesia scores for each EEG epoch.
    """
    BSRmap, BSR = suppression(eeg, Fs, stride)
    components = log_power_ratios(eeg, Fs, stride, BSRmap)
    depth_of_anesthesia = mixer(components, BSR)
    return depth_of_anesthesia


def suppression(
    eeg: np.ndarray, Fs: int = 128, stride: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determines whether an EEG segment is burst-suppressed and calculates BSR.

    Args:
        eeg (np.ndarray): The input EEG data as a column vector (in uV).
        Fs (int): The EEG sampling frequency (must be 128 Hz).
        stride (float): The stride value (0.5 seconds).

    Returns:
        Tuple[np.ndarray, np.ndarray]: BSRmap and BSR.
    """
    N, n_stride = n_epochs(eeg, Fs, stride)
    BSRmap = np.zeros(N)
    for n in range(N):
        x = segment(eeg, n + 6.5, 2, n_stride)
        BSRmap[n] = np.all(np.abs(x - baseline(x)) <= 5)
    BSR = (
        100
        * pd.Series(BSRmap).rolling(window=int((63 / stride) - 1), min_periods=1).mean()
    )
    return BSRmap, BSR


def log_power_ratios(eeg, Fs, stride, BSRmap):
    N, n_stride = n_epochs(eeg, Fs, stride)
    B, A = butter(2, 0.65 / (Fs / 2), "high")
    eeg_hi_pass_filtered = filtfilt(B, A, eeg)
    psd = np.empty((N, 4 * n_stride // 2)) * np.nan
    suppression_filter = piecewise(np.arange(0, 63.5, 0.5), [0, 3, 6], [0, 0.25, 1])
    components = np.empty((N, 3)) * np.nan

    for n in range(N):
        if is_not_burst_suppressed(BSRmap, n, 4):
            psd[n, :] = power_spectral_density(
                segment(eeg_hi_pass_filtered, n + 4, 4, n_stride)
            )
            # if sawtooth_detector(segment(eeg, n + 4, 4, n_stride), n_stride):
            #     psd[n, :] = suppression_filter * psd[n, :]

    for n in range(N):
        thirty_sec = time_range(30, n, stride)
        vhigh_power_conc = np.sqrt(
            np.mean(
                psd[thirty_sec, np.newaxis, band_range(39.5, 46.5, 0.5)]
                * psd[thirty_sec, np.newaxis, band_range(40, 47, 0.5)],
                axis=2,
            )
        )
        whole_power_conc = np.sqrt(
            np.mean(
                psd[thirty_sec, np.newaxis, band_range(0.5, 46.5, 0.5)]
                * psd[thirty_sec, np.newaxis, band_range(1, 47, 0.5)],
                axis=2,
            )
        )
        mid_band_power = prctmean(
            np.nanmean(10 * np.log10(psd[thirty_sec, band_range(11, 20, 0.5)]), axis=1),
            50,
            100,
        )
        components[n, 0] = (
            mean_band_power(psd[thirty_sec, :], 30, 47, 0.5) - mid_band_power
        )
        components[n, 1] = trim_mean(
            10 * np.log10(vhigh_power_conc / whole_power_conc), 0.5
        )
        components[n, 2] = (
            mean_band_power(psd[thirty_sec, :], 0.5, 4, 0.5) - mid_band_power
        )

    return components


def power_spectral_density(x):
    f = np.fft.fft(blackman(len(x)) * (x - baseline(x)))
    y = 2 * np.abs(f[: len(x) // 2]) ** 2 / (len(x) * np.sum(blackman(len(x)) ** 2))
    return y


def sawtooth_detector(eeg: np.ndarray, n_stride: int) -> bool:
    """
    Determines if this EEG segment contains a strong sawtooth-shaped K-complex.

    Args:
        eeg (np.ndarray): EEG data segment.
        n_stride (int): Number of samples per stride.

    Returns:
        bool: True if strong sawtooth-shaped K-complexes are detected, False otherwise.
    """
    saw = np.concatenate((np.zeros(n_stride - 5), np.arange(1, 6)))
    saw = (saw - np.mean(saw)) / np.std(saw)
    r = np.arange(1, len(eeg) - len(saw) + 1)
    v = np.convolve(eeg, np.ones(len(saw)), mode="valid")
    m1 = np.convolve(eeg, np.flip(saw), mode="valid") / len(saw)
    m2 = np.convolve(eeg, saw, mode="valid") / len(saw)
    m = np.vstack((m1, m2)).T
    v_broadcasted = np.broadcast_to(v[r - 1].reshape(-1, 1), m.shape)
    Y = np.max((v_broadcasted > 10) * (m / v_broadcasted), axis=1) > 0.63

    return np.any(Y)


def mixer(components, BSR):
    sedation_score = s_curve(components[:, 0], 104.4, 49.4, -13.9, 5.29)
    general_score = piecewise(components[:, 1], [-60.89, -30], [-40, 43.1])
    general_score = general_score + s_curve(
        components[:, 1], 61.3, 72.6, -24.0, 3.55
    ) * (components[:, 1] >= -30)
    bsr_score = piecewise(BSR, [0, 100], [50, 0])
    general_weight = piecewise(components[:, 2], [0, 5], [0.5, 1]) * (
        general_score < sedation_score
    )
    bsr_weight = piecewise(BSR, [10, 50], [0, 1])
    x = (sedation_score * (1 - general_weight)) + (general_score * general_weight)
    y = (
        piecewise(x, [-40, 10, 97, 110], [0, 10, 97, 100]) * (1 - bsr_weight)
        + bsr_score * bsr_weight
    )
    return y
