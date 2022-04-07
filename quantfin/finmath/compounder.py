from quantfin.calendars import DayCounts
from numpy import inf, exp, log, nan, empty, broadcast_arrays


class Compounder(object):
    _FREQ = ['day', '14-day', '28-day', '42-day', '84-day', '91-day',
             '182-day', '4-week', 'month', 'quarter', 'semi-annual',
             'annual', 'continuous', 'linear']
    __dc = None
    __freq = None

    def __init__(self, dc=None, freq='linear', adj=None, calendar='standard',
                 weekmask='Mon Tue Wed Thu Fri', adjoffset=0):
        self.dc = DayCounts(dc, adj=adj, calendar=calendar,
                            weekmask=weekmask, adjoffset=adjoffset)
        self.frequency = freq

    def simple_discount(self, r, d1, d2):
        """Calculate the discount factor for rate r between dates d1 and d2

        r is ZERO rate

        The word simple here denotes that only two dates are taken into
        account (hence only zero rates) and not spreads or percentage
        factors are taken into account
        """
        return 1 / self.simple_factor(r, d1, d2)

    def simple_factor(self, r, d1, d2):
        """Calculate simple factor base for rate r between dates d1 and d2

        r is ZERO (rate)

        The word simple here denotes that only two dates are taken into
        account (hence only zero rates) and no spreads or percentage
        factors are taken into account
        """
        tf = self.dc.tf(d1, d2)
        if self.frequency == 'linear':
            return 1 + r * tf
        elif self.frequency == 'continuous':
            return exp(r * tf)
        else:
            m = self.m(d1, d2)
            return (1 + r / m) ** (m * tf)

    def invc(self, factor, d1, d2):
        """Given a future value factor and two dates, function will inverse
        compound the implied rate for the frequency and day count specified
        in the class"""
        tf = self.dc.tf(d1, d2)
        if isinstance(tf, float):
            if abs(tf) < 1e-15:
                return nan
            elif self.frequency == 'simple':
                return (factor - 1) / tf
            elif self.frequency == 'continuous':
                return log(factor) / tf
            else:
                m = self.m(d1, d2)
                return m * (factor ** (1 / (m * tf)) - 1)
        else:
            mask = abs(tf) > 1e-15
            r = empty(len(tf)) * nan
            tf, factor = broadcast_arrays(tf, factor)
            if self.frequency == 'simple':
                r[mask] = (factor[mask] - 1) / tf[mask]
            elif self.frequency == 'continuous':
                r[mask] = log(factor[mask]) / tf[mask]
            else:
                m = self.m(d1, d2)
                r[mask] = m * (factor[mask] ** (1 / (m * tf[mask])) - 1)
            return r

    def convr(self, r, d1, d2, dc=None, freq='linear', adj=None,
              calendar='standard', weekmask='Mon Tue Wed Thu Fri',
              adjoffset=0):
        """Convert rate to a different frequency and day count.

        This is equivalent to calculate the FVF using the current rate
        conventions and then inverse compounding to the new one."""
        if dc is None:
            dc = self.dc.dc
        if adj is None:
            adj = self.dc.adj
        if calendar is None:
            calendar = self.dc.calendar

        factor = self.simple_factor(r, d1, d2)
        comp = Compounder(dc=dc, freq=freq, adj=adj, calendar=calendar,
                          weekmask=weekmask, adjoffset=adjoffset)
        return comp.invc(factor, d1, d2)

    def m(self, d1, d2):
        """M is the multiplier, the compounding frequency, used when
        producing discounts"""

        # Some multipliers are fixed. Go for the first
        mdict = {'4-week': 13,
                 'month': 12,
                 'quarter': 4,
                 'semi-annual': 2,
                 'annual': 1,
                 'continuous': inf,
                 'linear': None}
        try:
            return mdict[self.frequency]
        except KeyError:
            pass
        # Avoid unnecessary divisions
        if self.frequency == 'day':
            return self.dc.dib(d1, d2)
        base = {'14-day': 14,
                '28-day': 28,
                '42-day': 42,
                '84-day': 84,
                '91-day': 91,
                '182-day': 182}
        return self.dc.dib(d1, d2) / base[self.frequency]

    @property
    def dc(self):
        return self.__dc

    @dc.setter
    def dc(self, x):
        assert isinstance(x, DayCounts), 'Day count must be an object of ' \
                                         'type DayCounts.'
        self.__dc = x

    @property
    def frequency(self):
        return self.__freq

    @frequency.setter
    def frequency(self, x):
        assert isinstance(x, str), 'Frequency must be specified as a string'
        x = x.lower()
        msg = 'Unkown frequency type. Frequency must be one of: %s' \
              % ", ".join(self.supported_frequencies)
        assert x in self.supported_frequencies, msg
        self.__freq = x

    @property
    def supported_frequencies(self):
        return [x.lower() for x in self._FREQ]

    @supported_frequencies.setter
    def supported_frequencies(self, x):
        raise AttributeError('User may not set the list of supported '
                             'frequencues')
