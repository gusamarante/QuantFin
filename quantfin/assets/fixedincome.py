# TODO assert the format of 'data'
# TODO single date using a pandas series
# TODO Interpolations: flatforward
# TODO Forward Curve
# TODO Duration, Convexity, DV01, disccount
# TODO Chart

from scipy.interpolate import interp1d
from pandas import DataFrame, DatetimeIndex
from quantfin.calendar import DayCounts


class ZeroCurve(object):
    interp_methods = ['linear', 'cubic']

    def __init__(self, data, conv, interp='linear'):
        self._basic_assertions(data, conv, interp)

        self.data = data
        self.conv = conv
        self.interp = interp

    def rate(self, mat=None, date=None):
        # TODO deal with NAs in intermediary values of the curve
        # TODO assert mat is in the domain, type numeric
        # TODO extrapolation rule
        if date is None:
            rates = self.data.iloc[-1].values
            mats = self.data.iloc[-1].index.values
        else:
            rates = self.data.loc[date].values
            mats = self.data.loc[date].index.values

        fun = interp1d(mats, rates, kind=self.interp)

        if mat is None:
            return fun
        else:
            return fun(mat)

    @staticmethod
    def _basic_assertions(data, conv, interp):

        assert isinstance(data, DataFrame), 'data must be a pandas DataFrame'
        assert isinstance(data.index, DatetimeIndex), 'DataFrame index must be a DateTimeIndex'
        assert data.columns.is_numeric(), 'DataFrame columns must be numeric'

        assert conv.upper() in DayCounts.dc_domain(), f'Day count convention {conv} is not available'

        assert interp in ZeroCurve.interp_methods, 'Interpolation method not available'
