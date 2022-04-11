import pandas as pd


def compute_eri(total_return_index, funding_return):
    """
    Computes the excess returns indexes based on several total return indexes and one series of
    funding returns (risk-free rate). For now, there are no resample adjustments, so both of the
    frequencies must be in their correct timing and measurement.
    :param total_return_index: pandas.DataFrame with the total return indexes
    :param funding_return: pandas.Series with the funding returns
    :return: pandas.DataFrame where each column is the Excess Return Index for that given total return index.
    """
    assert isinstance(funding_return, pd.Series), 'funding returns must be a pandas.Series'

    total_returns = total_return_index.pct_change(1, fill_method=None)
    er = total_returns.subtract(funding_return, axis=0)
    er = er.dropna(how='all')

    eri = (1 + er).cumprod()
    eri = 100 * eri / eri.fillna(method='bfill').iloc[0]

    return eri
