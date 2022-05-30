from quantfin.portfolio import MaxSharpe, HRP, ERC
from tqdm import tqdm
import pandas as pd


class BacktestMaxSharpe(object):

    # TODO Documentation

    def __init__(self, eri, expected_returns, cov, risk_free, rebalance_dates,
                 short_sell=False, risk_aversion=None, name='Max Sharpe'):

        # TODO assert types, cov is multiindex, expected returns is DataFrame
        self.weights = pd.DataFrame(columns=eri.columns)
        self.expected_return = pd.Series(name='Expected Return', dtype=float)
        self.expected_vol = pd.Series(name='Expected Vol', dtype=float)

        for date in tqdm(rebalance_dates, name):

            # check if parameters are both available for this date, otherwise continue
            try:
                sigma = cov.xs(date)
                mu = expected_returns.loc[date].dropna()
            except KeyError:
                continue

            sigma = sigma.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if len(mu) == 0:
                continue

            # Select only assets that are available for both parameters
            available_assets = list(set(mu.index).intersection(set(sigma.index)))
            sigma = sigma.loc[available_assets, available_assets]
            mu = mu.loc[available_assets]
            rf = risk_free.loc[date]

            mkw = MaxSharpe(mu=mu, cov=sigma, rf=rf, risk_aversion=risk_aversion,
                            short_sell=short_sell)

            self.weights.loc[date] = mkw.risky_weights
            self.expected_return.loc[date] = mkw.mu_p
            self.expected_vol.loc[date] = mkw.sigma_p

        self.weights = self.weights.resample('D').last().fillna(method='ffill')
        self.weights = self.weights.reindex(eri.index, method='pad')

        self.expected_return = self.expected_return.resample('D').last().fillna(method='ffill')
        self.expected_return = self.expected_return.reindex(eri.index, method='pad')

        self.expected_vol = self.expected_vol.resample('D').last().fillna(method='ffill')
        self.expected_vol = self.expected_vol.reindex(eri.index, method='pad')

        return_index = (self.weights * eri.pct_change(1, fill_method=None))
        return_index = return_index.sum(axis=1, min_count=1).dropna()
        return_index = (1 + return_index).cumprod()
        return_index = 100 * return_index / return_index.iloc[0]
        self.return_index = return_index.rename(name)


class BacktestHRP(object):

    # TODO Documentation

    def __init__(self, eri, cov, rebalance_dates, method='single', metric='euclidean', name='HRP'):

        # TODO assert types, cov is multiindex, expected returns is DataFrame
        self.weights = pd.DataFrame(columns=eri.columns)

        for date in tqdm(rebalance_dates, name):

            # check if parameters are both available for this date, otherwise continue
            try:
                sigma = cov.xs(date)
            except KeyError:
                continue

            sigma = sigma.dropna(how='all', axis=0).dropna(how='all', axis=1)

            if sigma.empty:
                continue

            hrp = HRP(cov=sigma, method=method, metric=metric)

            self.weights.loc[date] = hrp.weights

        self.weights = self.weights.resample('D').last().fillna(method='ffill')
        self.weights = self.weights.reindex(eri.index, method='pad')

        return_index = (self.weights * eri.pct_change(1, fill_method=None))
        return_index = return_index.sum(axis=1, min_count=1).dropna()
        return_index = (1 + return_index).cumprod()
        return_index = 100 * return_index / return_index.iloc[0]
        self.return_index = return_index.rename(name)


class BacktestERC(object):

    # TODO Documentation

    def __init__(self, eri, cov, rebalance_dates, name='HRP'):

        self.weights = pd.DataFrame(columns=eri.columns)

        for date in tqdm(rebalance_dates, name):

            # check if parameters are both available for this date, otherwise continue
            try:
                sigma = cov.xs(date)
            except KeyError:
                continue

            sigma = sigma.dropna(how='all', axis=0).dropna(how='all', axis=1)

            if sigma.empty:
                continue

            erc = ERC(cov=sigma)

            self.weights.loc[date] = erc.weights

        self.weights = self.weights.resample('D').last().fillna(method='ffill')
        self.weights = self.weights.reindex(eri.index, method='pad')

        return_index = (self.weights * eri.pct_change(1, fill_method=None))
        return_index = return_index.sum(axis=1, min_count=1).dropna()
        return_index = (1 + return_index).cumprod()
        return_index = 100 * return_index / return_index.iloc[0]
        self.return_index = return_index.rename(name)
