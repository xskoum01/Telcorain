from builtins import range
import numpy as np
import pandas as pd

from numba import jit
from pycomlink.processing.baseline import _numba_baseline_constant
from pycomlink.processing.wet_antenna import _numba_waa_schleiss_2013


#from xarray_wrapper import xarray_apply_along_time_dim

#class Temperature:
"""
    #@xarray_apply_along_time_dim()
    def baseline_constant(temperature, wet, n_average_last_dry=1):

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
"""

@jit(nopython=True)
def _numba_waa_schleiss_2013(rsl, baseline, wet, waa_max, delta_t, tau):
    """Fast loop using numba to calculate WAA according to Schleiss et al 2013
    Parameters
    ----------
        rsl : iterable of float
                Time series of received signal level
        baseline : iterable of float
                Time series of baseline for rsl
        wet : iterable of int or iterable of float
               Time series with wet/dry classification information.
        waa_max : float
                  Maximum value of wet antenna attenuation
        delta_t : float
                  Parameter for wet antnenna attenation model
        tau : float
              Parameter for wet antnenna attenation model
    Returns
    -------
       iterable of float
           Time series of wet antenna attenuation
    """

    waa = np.zeros_like(rsl, dtype=np.float64)
    A = rsl - baseline

    for i in range(1, len(rsl)):
        if wet[i] == True:
            waa[i] = min(
                A[i], waa_max, waa[i - 1] + (waa_max - waa[i - 1]) * 3 * delta_t / tau
            )
        else:
            waa[i] = min(A[i], waa_max)
    return waa


#@xarray_apply_along_time_dim()
def waa_schleiss_2013(rsl, baseline, wet, waa_max, delta_t, tau):
    """Calculate WAA according to Schleiss et al 2013
    Parameters
    ----------
        rsl : iterable of float
                Time series of received signal level
        baseline : iterable of float
                Time series of baseline for rsl
        wet : iterable of int or iterable of float
               Time series with wet/dry classification information.
        waa_max : float
                  Maximum value of wet antenna attenuation
        delta_t : float
                  Parameter for wet antenna attention model
        tau : float
              Parameter for wet antenna attenuation model
    Returns
    -------
       iterable of float
           Time series of wet antenna attenuation
    Note
    ----
        The wet antenna adjusting is based on a peer-reviewed publication [1]_
    References
    ----------
    .. [1] Schleiss, M., Rieckermann, J. and Berne, A.: "Quantification and
                modeling of wet-antenna attenuation for commercial microwave
                links", IEEE Geoscience and Remote Sensing Letters, 10, 2013
    """

    waa = _numba_waa_schleiss_2013(
        rsl=np.asarray(rsl, dtype=np.float64),
        baseline=np.asarray(baseline, dtype=np.float64),
        #wet=np.asarray(wet, dtype=np.float64),
        wet=wet,
        waa_max=waa_max,
        delta_t=delta_t,
        tau=tau,
    )

    return waa
