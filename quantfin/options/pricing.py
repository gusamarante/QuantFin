from numpy import exp, sqrt, zeros, maximum


class BinomalTree(object):
    """
    Class that computes the Cox, Ross & Rubinstein (CRR) binomial tree model.
    """

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

        assert option_type in ['european', 'binary', 'american'], f"Option type '{option_type}' not implemented"
        assert type(n) is int, "'n' must be an integer"

        # Save inputs as attributes
        self.stock = stock
        self.strike = strike
        self.years2mat = years2mat
        self.vol = vol
        self.risk_free = risk_free
        self.div_yield = div_yield
        self.n = n
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

        callput = 1 if self.call else -1

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
