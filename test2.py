import asyncio

import numpy as np
import pandas as pd

from openibis.helpers import bandRange, timeRange
from openibis.openibis import openibis, suppression, gpt4suppresion

df = pd.read_csv("eeg.csv")
eeg = df["eeg"]
print(bandRange(40, 47, 0.5))
# thirty = timeRange(30, 1000, 0.5)
# print(thirty)
# print(len(thirty))
# print(n_epochs(eeg))
# # print(n_epochs(short_eeg))
# print(band_range(39.5, 46.5, 0.5))
# print(suppression(eeg))
# suppression(eeg[:64000], 128, 0.5)
# gpt4suppresion(eeg[:64000], 128, 0.5)

print(openibis(eeg[:64000]))
