import pandas as pd
import numpy as np
import asyncio
from openibis.helpers import n_epochs, piecewise
from openibis.openibis import suppression, log_power_ratios, openibis

df = pd.read_csv("eeg.csv")

# print(np.arange(0, 64, 0.5))
# print(piecewise(np.arange(0, 64, 0.5), [0,3,6], [0,0.25, 1]))

print(n_epochs(df["eeg"]))

# bsrmap, bsr = asyncio.run(suppression(df['eeg'], 0.5))
print(openibis(df["eeg"]))
