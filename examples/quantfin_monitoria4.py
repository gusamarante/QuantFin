import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
import scipy.optimize as opt
import scipy.cluster.hierarchy as sch
from scipy import stats


def cross_sectional_weights_from_signals(signals, weighting_scheme='rank', cov=None, vol_target=0.1):
    """
    This method calculates static long-short weights for a given set of signals
    Parameters
    ----------
    signals : a Pandas series containing a set of signals on which assets will be sorted. Typically, we want to
              be long and have higher weight on assets with large signals and to be short and have large negative
              weights in the assets with low signals
    weighting_scheme :  this is a string that can take the following values
        'zscores' : z-score long-short weights adding up to 200% in absolute value
        'winsorized' : same as 'zscores' but with z-scores winsorized at 10th/90th percentile limits
        'vol_target' : long-short weights set to achieve a certain volatility target for the entire portfolio
        'ERC' : Equal Risk Contribution Portfolio
        'IVP' : Inverse Volatility Portfolio
        'EW' : Equal Weights
        'rank' : Signal Rank Based Portfolio (this is the case if the parameter is not given or not recognized)
    cov : a DataFrame with the covariance matrix used in all weighting schemes but equal weights
    vol_target : used in the 'vol_target' and 'ERC' weighting schemes to set the overall volatility of the portfolio
    Returns
    -------
    a Pandas series with static long-short weights as type float
    """

    assert isinstance(signals, pd.Series), "input 'signals' must be a pandas Series"
    assert isinstance(weighting_scheme, str), "input 'weighting_scheme' must be a string"

    if weighting_scheme.lower().find('zscores')>-1:
        # z-score long-short weights adding up to 200% in absolute value
        weights = signals.copy().fillna(0) * 0
        scores = pd.Series(index=signals.dropna().index, data=stats.zscore(signals.dropna()))
        weights[scores.index] = scores.values
        weights = weights / (np.nansum(np.abs(weights)) / 2)

    elif weighting_scheme.lower().find('winsorized')>-1:
        # z-scores winsorized at 10th/90th percentile limits long-short weights adding up to 200%
        weights = signals.copy().fillna(0) * 0
        raw_scores = stats.zscore(signals.dropna())
        w_scores = stats.mstats.winsorize(raw_scores, limits=.1)
        scores = pd.Series(index=signals.dropna().index, data=w_scores)
        weights[scores.index] = scores.values
        weights = weights / (np.nansum(np.abs(weights)) / 2)

    elif weighting_scheme.lower().find('vol_target')>-1:
        # long-short weights set to achieve a certain volatility target for the entire portfolio

        # maximize the portfolio signal (actually minimize the opposite of that)
        port_signal = lambda x: - x.dot(signals.values)

        # subject to the portfolio volatility being equal to vol_target
        port_vol = lambda x: np.sqrt(x.dot(cov).dot(x)) - vol_target
        eq_cons = {'type': 'eq', 'fun': lambda x: port_vol(x)}

        # initialize optimization with rank-based portfolio
        ranks = signals.rank()
        w0 = ranks - ranks.mean()
        w0 = w0 / (np.nansum(np.abs(w0)) / 2)

        # bounds are set in order to be long/short what the rank based portfolio tells us to be long/short
        # the maximum weight in absolute value is the maximum weight in the rank-based portfolio
        bounds = pd.DataFrame(index=signals.index, columns=['lower', 'upper'])
        bounds['lower'] = np.array([np.sign(w0) * max(np.abs(w0)), np.zeros(w0.shape)]).min(axis=0)
        bounds['upper'] = np.array([np.sign(w0) * max(np.abs(w0)), np.zeros(w0.shape)]).max(axis=0)

        res = opt.basinhopping(port_signal, np.nan_to_num(w0.values),
                    minimizer_kwargs={'method': 'SLSQP', 'constraints': eq_cons, 'bounds': bounds.values},
                               T=1.0,
                               niter=500,
                               stepsize=0.5,
                               interval=50,
                               disp=False,
                               niter_success=100)

        if not res['lowest_optimization_result']['success']:
            raise ArithmeticError('Optimization convergence failed for volatility target weighting scheme')

        weights = pd.Series(index=signals.index, data = np.nan_to_num(res.x))

    elif weighting_scheme.find('ERC') > -1:
        # Equal Risk Contribution Portfolio

        # minimize the distance to the equal risk portfolio
        n = cov.shape[0]
        target_risk_contribution = np.ones(n) / n
        dist_to_target = lambda x: np.linalg.norm(x * (x @ cov / (vol_target ** 2)) - target_risk_contribution)

        # subject to the portfolio volatility being equal to vol_target
        port_vol = lambda x: np.sqrt(x.dot(cov).dot(x))
        eq_cons = {'type': 'eq', 'fun': lambda x: port_vol(x) - vol_target}

        # initialize optimization with rank-based portfolio
        ranks = signals.rank()
        w0 = ranks - ranks.mean()
        w0 = w0 / (np.nansum(np.abs(w0)) / 2)

        # bounds are set in order to be long/short what the rank based portfolio tells us to be long/short
        # the maximum weight in absolute value is the maximum weight in the rank-based portfolio
        bounds = pd.DataFrame(index=signals.index, columns=['lower', 'upper'])
        bounds['lower'] = np.array([np.sign(w0) * max(np.abs(w0)), np.zeros(w0.shape)]).min(axis=0)
        bounds['upper'] = np.array([np.sign(w0) * max(np.abs(w0)), np.zeros(w0.shape)]).max(axis=0)

        res = opt.basinhopping(dist_to_target, target_risk_contribution,
                               minimizer_kwargs={'method': 'SLSQP', 'constraints': eq_cons, 'bounds': bounds.values},
                               T=1.0,
                               niter=500,
                               stepsize=0.5,
                               interval=50,
                               disp=False,
                               niter_success=100)

        if not res['lowest_optimization_result']['success']:
            raise ArithmeticError('Optimization convergence failed for ERC weighting scheme')
        weights = pd.Series(index=signals.index, data=np.nan_to_num(res.x))

    elif weighting_scheme.find('IVP')>-1:
        # Inverse Volatility Portfolio
        ranks = signals.rank()
        weights = ranks - ranks.mean()
        vols = pd.Series(index=cov.index, data=np.sqrt(np.diag(cov)))
        weights = np.sign(weights) / vols
        weights = weights / (np.nansum(np.abs(weights)) / 2)

    elif weighting_scheme == 'EW':
        # Equal Weights
        ranks = signals.rank()
        weights = ranks - ranks.mean()
        weights = np.sign(weights) / signals.shape[0]
        weights = weights / (np.nansum(np.abs(weights)) / 2)

    else:
        # Signal Rank Based Portfolio
        if weighting_scheme.lower().find('rank') == -1:
            print('Unclear weighting scheme, assuming signal-rank based weights')
        ranks = signals.rank()
        weights = ranks - ranks.mean()
        weights = weights / (np.nansum(np.abs(weights)) / 2)

    return weights.astype(float)


file_path = r'/Users/gusamarante/Dropbox/Aulas/Insper - Financas Quantitativas/2022/Monitoria 1/fx_data.xlsx'
df_trackers = pd.read_excel(file_path, index_col=0, sheet_name='trackers')
df_carry = pd.read_excel(file_path, index_col=0, sheet_name='carry')
df_value = pd.read_excel(file_path, index_col=0, sheet_name='ppp_value')

countries2keep = ['AUD', 'BRL', 'CAD', 'CHF', 'CLP', 'CNH', 'CZK', 'EUR', 'GBP', 'HUF', 'IDR',
                  'ILS', 'INR', 'JPY', 'KRW', 'MXN', 'NOK', 'NZD', 'PLN', 'RUB', 'SEK', 'TRY', 'ZAR']

df_trackers = df_trackers[countries2keep]
df_carry = df_carry[countries2keep]
df_value = df_value[countries2keep]

covmat = df_trackers.resample('M').last().pct_change(1).cov() * 12
current_signal = df_value.iloc[-1]

current_weights = cross_sectional_weights_from_signals(signals=current_signal,
                                                       weighting_scheme='vol_target',
                                                       cov=covmat,
                                                       vol_target=0.2)

current_weights.dropna().to_clipboard()


