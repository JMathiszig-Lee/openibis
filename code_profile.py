import cProfile
import pstats

import pandas as pd

from openibis.openibis import suppression, vector_suppression

df = pd.read_csv("eeg.csv")
eeg = df["eeg"]

# Profile the original function
profile = cProfile.Profile()
profile.enable()
suppression(eeg)
profile.disable()
stats = pstats.Stats(profile).sort_stats("tottime")
stats.print_stats()

# Profile the Numba-optimized function
profile_numba = cProfile.Profile()
profile_numba.enable()
vector_suppression(eeg)
profile_numba.disable()
stats_numba = pstats.Stats(profile_numba).sort_stats("tottime")
stats_numba.print_stats()
