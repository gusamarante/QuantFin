"""
Classes to evaluate portfolio performance
- Performance Table
    - Annualized Return
    - Annualized Standard Deviation
    - Sharpe Index
    - Calmar Rotio
    - Sortino
    - Expected Shortfall (Historical)
    - Value at Risk (Historical)
    - Maximum Drawdown
- Drawdown calculator
- Charts
    - Total Return Index
    - Underwater Chart
    - Histogram with Stats
    - Rolling Sharpe
"""


class Perfomance(object):

    def __init__(self, total_return):
        self.total_return = total_return
        self.returns = total_return.pct_change(1)
        self.drawdowns = self._get_drawdowns()

    def _get_drawdowns(self):
        exp_max = self.total_return.expanding().max()
        a = 1
