from numpy import exp, sqrt, zeros, maximum, log, abs
import matplotlib.pyplot as plt
from scipy.stats import norm


class BinomalTree(object):
    """
    Class that computes the Cox, Ross & Rubinstein (CRR) binomial tree model.
    """

    implemented_types = ['european', 'binary', 'american']

    def __init__(self, stock, strike, years2mat, vol, risk_free=0, div_yield=0,
                 n=1, call=True, option_type='european'):
        """

        :param stock: float, current stock price
        :param strike: float, strike price of the option
        :param years2mat: float, years until maturity
        :param vol: float, estimate of the stock return volatility
        :param risk_free: float, continously compounded risk-free interest rate for the period
        :param div_yield: float, continously compounded dividend yield of the stock
        :param n: int, number of periods in the binomial tree
        :param call: bool, call option if true, put option if false
        :param option_type: str, 'european', 'american' (allows for early exercise) or 'binary' (also called "digital")
        """

        assert option_type in self.implemented_types, f"Option type '{option_type}' not implemented"

        # Save inputs as attributes
        self.stock = stock
        self.strike = strike
        self.years2mat = years2mat
        self.vol = vol
        self.risk_free = risk_free
        self.div_yield = div_yield
        self.n = int(n)
        self.call = call
        self.option_type = option_type

        # compute other attributes
        self.dt = years2mat / n
        self.df = exp(-risk_free * self.dt)
        self.up_factor = exp(self.vol * sqrt(self.dt))
        self.down_factor = 1 / self.up_factor
        self.rn_prob = (exp((risk_free - div_yield) * self.dt) - self.down_factor) / \
                       (self.up_factor - self.down_factor)

        self.tree_stock = self._get_stock_tree()
        self.tree_option, self.delta = self._get_option_tree()
        self.price = self.tree_option[0, 0]

    def _get_stock_tree(self):
        tree = zeros((self.n + 1, self.n + 1))

        for j in range(0, self.n + 1):
            for i in range(j + 1):
                tree[i, j] = self.stock * (self.up_factor ** (j - i)) * (self.down_factor ** i)

        return tree

    def _get_option_tree(self):
        delta = zeros((self.n + 1, self.n + 1))
        option = zeros((self.n + 1, self.n + 1))

        for j in range(self.n + 1, 0, -1):
            for i in range(j):
                if j == self.n + 1:  # Terminal Payoff
                    option[i, j-1] = self._payoff(self.tree_stock[i, j-1],  self.strike)

                else:  # Inner nodes
                    pv = self.df * (self.rn_prob*option[i, j] + (1 - self.rn_prob)*option[i+1, j])

                    if self.option_type == 'american':  # check early exercise
                        payoff = self._payoff(self.tree_stock[i, j - 1], self.strike)
                        option[i, j - 1] = maximum(pv, payoff)
                    else:
                        option[i, j - 1] = pv

                    delta[i, j-1] = (option[i, j] - option[i+1, j]) / (self.tree_stock[i, j] - self.tree_stock[i+1, j])

        return option, delta[0, 0]

    def _payoff(self, stock, strike):
        callput = 1 if self.call else -1

        if self.option_type == 'binary':
            if self.call:
                payoff = 1 if stock > strike else 0
            else:
                payoff = 1 if stock < strike else 0

        else:
            payoff = maximum((stock - strike) * callput, 0)

        return payoff

    def chart_stock(self, labels=False):

        plt.figure()

        # Plot labels
        if labels:
            for j in range(self.n + 1):
                for i in range(j + 1):
                    plt.text(j*self.dt, self.tree_stock[i, j], '    ' + str(round(self.tree_stock[i, j], 2)),
                             verticalalignment='center', horizontalalignment='left')

        # Plot edges
        for j in range(self.n):
            for i in range(j + 1):
                plt.plot([j * self.dt, (j + 1) * self.dt], [self.tree_stock[i, j], self.tree_stock[i, j + 1]],
                         marker='o', color='black')
                plt.plot([j * self.dt, (j + 1) * self.dt], [self.tree_stock[i, j], self.tree_stock[i + 1, j + 1]],
                         marker='o', color='black')

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlabel('Years')
        ax.set_ylabel('Price')

        plt.tight_layout()

        return ax.get_figure()


class BlackScholes(object):
    # TODO Documentation

    implemented_types = ['european', 'binary']  # TODO american, barriers

    def __init__(self, stock_price, strike_price, maturity, risk_free, vol, div_yield=0, call=True,
                 option_type='european'):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.maturity = maturity
        self.risk_free = risk_free
        self.vol = vol
        self.div_yield = div_yield
        self.call = call
        self.option_type = option_type
        self.d1 = (log(stock_price / strike_price) + maturity * (risk_free - div_yield + 0.5 * (vol ** 2))) / (
                vol * sqrt(maturity))
        self.d2 = self.d1 - vol * sqrt(maturity)

        nd1 = norm.cdf(self.d1)
        nd2 = norm.cdf(self.d2)
        nmd1 = norm.cdf(-self.d1)
        nmd2 = norm.cdf(-self.d2)
        nd1p = norm.pdf(self.d1)
        nd2p = norm.pdf(self.d2)

        if option_type == 'european':

            # price
            if call:
                self.price = (stock_price * exp(-div_yield * maturity) * nd1 - strike_price * exp(
                    - risk_free * maturity) * nd2)
            else:
                self.price = - (stock_price * exp(-div_yield * maturity) * nmd1 + strike_price * exp(
                    - risk_free * maturity) * nmd2)

            # delta
            if call:
                self.delta = exp(-div_yield * maturity) * nd1
            else:
                self.delta = exp(-div_yield * maturity) * (nd1 - 1)

            # gamma (same for european calls and puts)
            self.gamma = (exp(-div_yield * maturity) * nd1p) / (vol * stock_price * sqrt(maturity))

            # theta
            if call:
                self.theta = - (vol * stock_price * exp(-div_yield * maturity) * nd1p) / (
                        2 * sqrt(maturity)) + div_yield * stock_price * nd1 * exp(
                    -div_yield * maturity) - risk_free * strike_price * exp(-risk_free * maturity) * nd2
            else:
                self.theta = - (vol * stock_price * exp(-div_yield * maturity) * nd1p) / (
                        2 * sqrt(maturity)) - div_yield * stock_price * nmd1 * exp(
                    -div_yield * maturity) + risk_free * strike_price * exp(-risk_free * maturity) * nmd2

            # vega (same for european calls and puts)
            self.vega = stock_price * sqrt(maturity) * exp(-div_yield * maturity) * nd1p

            # rho
            if call:
                self.rho = strike_price * maturity * exp(-risk_free * maturity) * nd2
            else:
                self.rho = - strike_price * maturity * exp(-risk_free * maturity) * nmd2

        elif option_type == 'binary':

            # price
            if call:
                self.price = exp(-risk_free * maturity) * nd2
            else:
                self.price = exp(-risk_free * maturity) * (1 - nd2)

            # delta
            if call:
                self.delta = (exp(-risk_free * maturity) * nd2p) / (vol * stock_price * sqrt(maturity))
            else:
                self.delta = - (exp(-risk_free * maturity) * nd2p) / (vol * stock_price * sqrt(maturity))

            # gamma
            if call:
                self.gamma = - (exp(-risk_free * maturity) * self.d1 * nd2p) / (
                            (vol * stock_price * sqrt(maturity)) ** 2)
            else:
                self.gamma = (exp(-risk_free * maturity) * self.d1 * nd2p) / ((vol * stock_price * sqrt(maturity)) ** 2)

            # theta
            if call:
                self.theta = risk_free * exp(-risk_free * maturity) * nd2 + exp(-risk_free * maturity) * nd2p * (
                            self.d1 / (2 * maturity) - (risk_free - div_yield) / (vol * sqrt(maturity)))
            else:
                self.theta = risk_free * exp(-risk_free * maturity) * (1 - nd2) - exp(-risk_free * maturity) * nd2p * (
                            self.d1 / (2 * maturity) - (risk_free - div_yield) / (vol * sqrt(maturity)))

            # vega
            if call:
                self.vega = - exp(-risk_free * maturity) * nd2p * (sqrt(maturity) + self.d2 / vol)
            else:
                self.vega = exp(-risk_free * maturity) * nd2p * (sqrt(maturity) + self.d2 / vol)

            # rho
            if call:
                self.rho = -maturity * exp(-risk_free * maturity) * nd2 + (sqrt(maturity) / vol) * exp(
                    -risk_free * maturity) * nd2p
            else:
                self.rho = -maturity * exp(-risk_free * maturity) * (1 - nd2) - (sqrt(maturity) / vol) * exp(
                    -risk_free * maturity) * nd2p

    @classmethod
    def from_price(cls, stock_price, strike_price, maturity, risk_free, option_price, div_yield=0, call=True,
                   option_type='european', error=10e-8):
        """
        This is an alternative constructor for the BlackScholes class to find the implied volatility, given
        the observed price of the option. from observed prices of option.

        Notice that the input for volatility of swaped for the price of the option. The newton-rhapson method
        is applied to find the implied volatility of the option.
        """

        vol = 0.2  # initial guess
        dv = error + 1

        while abs(dv) > error:
            bs = BlackScholes(stock_price, strike_price, maturity, risk_free, vol, div_yield, call, option_type)
            imp_price = bs.price
            vega = bs.vega
            price_error = imp_price - option_price
            dv = price_error / vega
            vol = vol - dv

        bs = BlackScholes(stock_price, strike_price, maturity, risk_free, vol, div_yield, call, option_type)
        return bs
