from builtins import range
import numpy as np
import pandas as pd

from numba import jit

#from .xarray_wrapper import xarray_apply_along_time_dim

#@xarray_apply_along_time_dim()
def baseline_constant(temperature, wet, n_average_last_dry=1):
    """
    Build baseline with constant level during a `wet` period
    Parameters
    ----------
    temperature : numpy.array or pandas.Series
        Array of the temperatures of the units.
    wet : numpy.array or pandas.Series
        Information if classified index of times series is wet (True)
        or dry (False). Note that `NaN`s in `wet` will lead to `NaN`s in
        `baseline` also after the `NaN` period since it is then not clear
        whether or not there was a change of wet/dry within the `NaN` period.
    n_average_last_dry: int, default = 1
        Number of last baseline values before start of wet event that should
        be averaged to get the value of the baseline during the wet event.
        Note that this values should not be too large because the baseline
        might be at an expected level, e.g. if another wet event is
        ending shortly before.
    Returns
    -------
    baselineTemp : numpy.array
          Baseline during wet period
    """

    return _numba_baseline_constant(
        temperature=np.asarray(temperature, dtype=np.float64),
        wet=np.asarray(wet, dtype=np.bool),
        n_average_last_dry=n_average_last_dry,
    )


@jit(nopython=True)
def _numba_baseline_constant(temperature, wet, n_average_last_dry):
    baselineTemp = np.zeros_like(temperature, dtype=np.float64)
    baselineTemp[0:n_average_last_dry] = temperature[0:n_average_last_dry]
    for i in range(n_average_last_dry, len(temperature)):
        if np.isnan(wet[i]):
            baselineTemp[i] = np.NaN
        elif wet[i] & ~wet[i - 1]:
            baselineTemp[i] = np.mean(baselineTemp[(i - n_average_last_dry): i])
        elif wet[i] & wet[i - 1]:
            baselineTemp[i] = baselineTemp[i - 1]
        else:
            baselineTemp[i] = temperature[i]
    return baselineTemp
