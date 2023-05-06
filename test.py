import pandas as pd
import numpy as np
import asyncio
from openibis.helpers import number_of_epochs, piecewise
from openibis.openibis import suppression, log_power_ratios

df = pd.read_csv("eeg.csv")

# print(np.arange(0, 64, 0.5))
# print(piecewise(np.arange(0, 64, 0.5), [0,3,6], [0,0.25, 1]))

print(number_of_epochs(df["eeg"]))

# bsrmap, bsr = asyncio.run(suppression(df['eeg'], 0.5))

bsrmap = np.ones(1000000)
bsrmap = bsrmap > 2
print(bsrmap)
components = asyncio.run(log_power_ratios(df["eeg"], 0.5, bsrmap))
