from scipy.interpolate import interp1d
from pandas import DataFrame, DatetimeIndex, to_datetime, date_range
from pandas.tseries.offsets import MonthEnd
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
        q = exp(-integral[0])
        return q


class CDS(object):
    # TODO Premium leg cash flow
    # TODO Accrued Premium
    # TODO Protection leg cash flow
    # TODO RPV01

    def __init__(self, contract_spread, effective_date, maturity, notional=100, conv='ACT/360'):

        self.contract_spread = contract_spread
        self.effective_date = to_datetime(effective_date)
        self.maturity = to_datetime(maturity)  # TODO maturity could be a datetime or a timedelta like '5Y'
        self.notional = notional
        self.conv = conv
        self.premium_cf = self._premium_leg_cf()

    def _premium_leg_cf(self):
        start = self.effective_date.replace(day=1)
        end = self.maturity + MonthEnd(0)
        premium_dates = date_range(start, end, freq='QS-DEC')
        premium_dates = premium_dates.shift(19, freq='D')

        cond1 = premium_dates >= self.effective_date
        cond2 = premium_dates <= self.maturity
        premium_dates = premium_dates[cond1 & cond2]

        df = DataFrame(index=premium_dates)

        # pd.date_range('2020-09-16', '2025-09-16', freq='Q').shift(20, freq='D')
        return premium_dates
