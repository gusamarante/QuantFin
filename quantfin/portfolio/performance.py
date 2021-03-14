"""
Classes to evaluate portfolio performance
- Performance Table
    - Sortino
    - Expected Shortfall (Historical)
    - Value at Risk (Historical)
- Charts
    - Total Return Indexes
    - Underwater Chart
    - Histogram with Stats
    - Rolling Sharpe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perfomance(object):
    # TODO Notebook example

    def __init__(self, total_return):
        # TODO Documentation

        self.total_return = total_return
        self.returns_ts = total_return.pct_change(1)
        self.returns_ann = self._get_ann_returns()
        self.std = self._get_ann_std()
        self.sharpe = self.returns_ann / self.std
        self.skewness = self.returns_ts.skew()
        self.kurtosis = self.returns_ts.kurtosis()  # Excess Kurtosis (k=0 is normal)
        self.sortino = self._get_sortino()
        self.drawdowns = self._get_drawdowns()
        self.max_dd = self.drawdowns.groupby(level=0).min()['dd']

    def plot_drawdowns(self, name, n=5):
        # TODO Make it neater

        plt.figure()
        plt.plot(self.total_return[name], zorder=-1, color='blue')

        for dd in range(n):
            start = self.drawdowns.loc[name].loc[dd, 'last start']
            end = self.drawdowns.loc[name].loc[dd, 'end']
            plt.plot(self.total_return[name].loc[start: end], color='red')

        plt.show()

    def _get_drawdowns(self):
        # TODO Documentaion and explanations
        # TODO BUG: If the latest value is a drawdown, it does not get into the list

        df_drawdowns = pd.DataFrame()

        for col in self.total_return.columns:
            data = pd.DataFrame(index=self.total_return.index)
            data['tracker'] = self.total_return[col]
            data['expanding max'] = data['tracker'].expanding().max()
            data['dd'] = data['tracker'] / data['expanding max'] - 1
            data['iszero'] = data['dd'] == 0
            data['current min'] = 0
            data['last start'] = data.index[0]
            data['end'] = data.index[0]

            # Find the rolling current worst drawdown
            for date, datem1 in zip(data.index[1:], data.index[:-1]):
                if data.loc[date, 'iszero']:
                    data.loc[date, 'current min'] = 0
                elif data.loc[date, 'dd'] < data.loc[datem1, 'current min']:
                    data.loc[date, 'current min'] = data.loc[date, 'dd']
                else:
                    data.loc[date, 'current min'] = data.loc[datem1, 'current min']

            # find the last start of the drawdown
            for date, datem1 in zip(data.index[1:], data.index[:-1]):
                if data.loc[date, 'iszero']:
                    data.loc[date, 'last start'] = date
                else:
                    data.loc[date, 'last start'] = data.loc[datem1, 'last start']

            # find the end of each drawdown
            for date, datem1 in zip(data.index[1:], data.index[:-1]):
                if data.loc[date, 'current min'] < data.loc[datem1, 'current min']:
                    data.loc[date, 'end'] = date
                else:
                    data.loc[date, 'end'] = data.loc[datem1, 'end']

            # find drawdown splits
            data['dd duration'] = (data['end'] - data['last start']).dt.days
            data['dd duration shift'] = data['dd duration'].shift(-1)
            data['isnegative'] = data['dd duration shift'] < 0

            data = data.reset_index()

            data = data[data['isnegative']]

            data = data.sort_values('current min', ascending=True)

            data = data[data['current min'] < 0]

            data = data[['current min', 'last start', 'end']].reset_index().drop('index', axis=1)

            df_add = pd.DataFrame(index=pd.MultiIndex.from_product([[col], data.index]),
                                  data=data.values,
                                  columns=['dd', 'last start', 'end'])

            df_drawdowns = df_drawdowns.append(df_add)

        return df_drawdowns

    def _get_ann_returns(self):

        df_ret = pd.Series(name='Annualized Returns')
        for col in self.total_return.columns:
            aux = self.total_return[col].dropna()
            start, end = aux.iloc[0], aux.iloc[-1]
            n = len(aux) - 1
            df_ret.loc[col] = (end / start) ** (252 / n) - 1  # TODO adjustment factor goes here

        return df_ret

    def _get_ann_std(self):

        df_std = pd.Series(name='Annualized Standard Deviation')
        for col in self.total_return.columns:
            aux = self.returns_ts[col].dropna()
            df_std.loc[col] = aux.std() * np.sqrt(252)  # TODO adjustment factor goes here

        return df_std

    def _get_sortino(self):

        df_sor = pd.Series(name='Sortino')
        for col in self.total_return.columns:
            aux = self.returns_ts[col][self.returns_ts[col] < 0].dropna()
            df_sor.loc[col] = self.returns_ann[col] / (np.sqrt(252) * aux.std())  # TODO adjustment factor goes here

        return df_sor
