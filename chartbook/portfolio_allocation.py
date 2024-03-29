from scipy.optimize import minimize, Bounds, LinearConstraint
from quantfin.data import tracker_feeder, DROPBOX, SGS
from quantfin.statistics import GaussianHMM
from quantfin.portfolio import ERC, HRP
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# Excel file to save outputs
writer = pd.ExcelWriter(DROPBOX.joinpath(f'Allocation Regimes.xlsx'))

# Read Trackers
df_tri = tracker_feeder()
df_tri = df_tri[['Pillar Equity Brazil', 'Pillar Equity Global',
                 'Pillar Nominal Rate', 'Pillar Real Rate']]

# Macro Data
sgs = SGS()

ipca = sgs.fetch({433: 'IPCA'})
ipca = ipca.resample('M').last()
ipca = (1 + ipca/100).cumprod()
ipca = ipca.resample('Q').last().pct_change()


pib = sgs.fetch({24364: 'IBCBR'})
pib = pib.resample('Q').mean().pct_change()

# ===============================
# ===== Hidden Markov Model =====
# ===============================
input_data = df_tri.resample('Q').last().drop([], axis=1).pct_change().dropna()
# input_data = pd.concat([input_data, pib, ipca], axis=1).dropna()
hmm = GaussianHMM(returns=input_data)
hmm.fit(n_states=3, fit_iter=200)

hmm.trans_mat.to_excel(writer, 'Transition')
hmm.means.to_excel(writer, 'Mean')
hmm.vols.to_excel(writer, 'Vols')
(hmm.means / hmm.vols).to_excel(writer, 'Sharpe')
hmm.corrs.to_excel(writer, 'Correlations')
hmm.state_probs.to_excel(writer, 'State Probs')

# Optimal allocation in each state
df_weights = pd.DataFrame(columns=hmm.means.columns)
df_port = pd.DataFrame()

for state in hmm.means.index:
    mu = hmm.means.loc[state]
    cov = hmm.covars.loc[state]
    n_assets = len(mu)

    def objfun(x):
        v = np.sqrt(x @ cov @ x)
        r = x @ mu
        s = r / v
        return -s

    w0 = np.ones(n_assets) * (1 / n_assets)
    bounds = Bounds(np.zeros(n_assets), np.ones(n_assets))
    cons_weight = LinearConstraint(A=np.ones(n_assets),
                                   lb=1, ub=1)
    res = minimize(objfun, w0, bounds=bounds, constraints=cons_weight)

    df_weights.loc[state] = res.x
    df_port.loc[state, 'Mean'] = res.x @ mu
    df_port.loc[state, 'Vol'] = np.sqrt(res.x @ cov @ res.x)

df_port['Sharpe'] = df_port['Mean'] / df_port['Vol']
df_port.to_excel(writer, 'State Portfolios')
df_weights.to_excel(writer, 'Weights by state')

# ===================================
# ===== Equal Risk Contribution =====
# ===================================
cov_m = df_tri.resample('M').last().pct_change(1).dropna().cov()
cov_q = df_tri.resample('Q').last().pct_change(1).dropna().cov()
erc_m = ERC(cov=cov_m, bounded=True)
erc_q = ERC(cov=cov_q, bounded=True)

erc_weights = pd.concat([erc_q.weights.rename('Quarterly'), erc_m.weights.rename('Monthly')], axis=1)
erc_weights.to_excel(writer, 'ERC Weights')

# ====================================
# ===== Hierarchical Risk Parity =====
# ====================================
hrp_m = HRP(cov_m)
hrp_q = HRP(cov_q)
hrp_weights = pd.concat([hrp_q.weights.rename('Quarterly'), hrp_m.weights.rename('Monthly')], axis=1)
hrp_weights.to_excel(writer, 'HRP Weights')


writer.save()
