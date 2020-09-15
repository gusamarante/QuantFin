from scipy.interpolate import interp1d
from pandas import DataFrame, DatetimeIndex
from quantfin.calendar import DayCounts
from scipy.integrate import quad
from numpy import exp


class ZeroCurve(object):
    # TODO assert the format of 'data'
    # TODO single date using a pandas series
    # TODO Interpolations: flatforward
    # TODO Duration, Convexity, DV01
    # TODO Forward Curve, Carry
    # TODO Chart
    # TODO grab time series of a single maturity

    interp_methods = ['linear', 'cubic']

    def __init__(self, data, conv, interp='linear'):
        self._basic_assertions(data, conv, interp)

        self.data = data
        self.conv = conv
        self.interp = interp
        self.dc = DayCounts(conv)
        self.y_base = self.dc.dib()

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

    def discount(self, mat, date=None):
        # TODO think of a way to return a disccount function when mat=None
        r = self.rate(mat, date)
        d = 1/((1+r)**(mat/self.y_base))
        return d

    def forward(self, mat1, mat2, date=None):
        assert mat1 < mat2, 'mat1 must be smaller than mat2'
        r1 = self.rate(mat1, date)
        r2 = self.rate(mat2, date)
        fwd = (((1+r2)**(mat2/self.y_base))/((1+r1)**(mat1/self.y_base)))**(self.y_base/(mat2 - mat1)) - 1
        return fwd

    @staticmethod
    def _basic_assertions(data, conv, interp):

        assert isinstance(data, DataFrame), 'data must be a pandas DataFrame'
        assert isinstance(data.index, DatetimeIndex), 'DataFrame index must be a DateTimeIndex'
        assert data.columns.is_numeric(), 'DataFrame columns must be numeric'

        assert conv.upper() in DayCounts.dc_domain(), f'Day count convention {conv} is not available'

        assert interp in ZeroCurve.interp_methods, 'Interpolation method not available'


class HazardRateTermStructure(object):

    def __init__(self):
        self.mathaz = {}
        self._first_entry = True

    def add_hazard(self, mat, prob):
        # TODO assert mat and prob
        # TODO add support of interable inputs

        if self._first_entry:
            self.mathaz[0] = prob
            self.mathaz[mat] = prob
            self._first_entry = False
        else:
            self.mathaz[mat] = prob

    def fwd_hazard(self, t):
        fun = interp1d(list(self.mathaz.keys()), list(self.mathaz.values()), kind='next')
        return fun(t)

    def survival_prob(self, t1, t2):
        # TODO assert t1, t2 are in the domain
        integral = quad(self.fwd_hazard, t1, t2)
        Q = exp(-integral[0])
        return Q
