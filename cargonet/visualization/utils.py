import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, interp1d
from scipy.signal import resample


def resample_time_series(times, values, samples_per_minute=30, smooth=True):
    tr = times # mdates.date2num(times)
    if smooth:
        assert 60.0 % samples_per_minute == 0
        start, end = min(times), max(times)
        try:
            duration = (end - start).total_seconds()
            tr = pd.date_range(start, end, freq="%ds" % int(60.0 / samples_per_minute))
            times = mdates.date2num(times)
        except Exception:
            duration = len(times)
            tr = times

        # Cut to same length
        l = min(len(times), len(values))
        times, values = times[:l], values[:l]

        # Resample and eventually apply b-spline smoothing
        samples = int(duration / 60.0 * samples_per_minute)
        _times = np.linspace(times.min(), times.max(), samples)
        if smooth:
            spl = make_interp_spline(times, values, k=3)
        else:
            spl = interp1d(times, values)
        values = spl(_times)
        times = _times

    # Cut to same length again
    l = min(len(tr), len(values))
    df = pd.DataFrame(dict(t=tr[:l], values=values[:l]), columns=["t", "values"])
    df = df.set_index(pd.DatetimeIndex(df["t"]))
    return df
