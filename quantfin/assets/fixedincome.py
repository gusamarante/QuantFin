# TODO assert the format of 'data'
# TODO single date using a pandas series
# TODO Interpolations: linear, flatforward, cubic spline
# TODO Forward Curve
# TODO Duration, Convexity, DV01, disccount
# TODO Chart

from scipy.interpolate import interp1d, interp2d


class ZeroCurve(object):

    def __init__(self, data, conv, interp='linear'):
        self.data = data
        self.conv = conv
        self.interp = interp

    def rate(self, mat=None, date=None):
        # TODO deal with NAs in intermediary values of the curve
        # TODO assert mat is in the domain, type numeric
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



        return rates
