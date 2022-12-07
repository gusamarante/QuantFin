import pandas as pd
from quantfin.statistics import GaussianHMM
from quantfin.data import tracker_feeder, DROPBOX, SGS, FRED

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# Excel file to save outputs
writer = pd.ExcelWriter(DROPBOX.joinpath(f'Allocation Regimes.xlsx'))

# Read Data
df_tri = tracker_feeder()
df_tri = df_tri[['Pillar Credito', 'Pillar Equity Brazil', 'Pillar Equity Global',
                 'Pillar FIP', 'Pillar Nominal Rate', 'Pillar Real Rate']]

# ===============================
# ===== Hidden Markov Model =====
# ===============================
input_returns = df_tri.resample('Q').last().drop(['Pillar FIP', 'Pillar Credito'], axis=1).pct_change().dropna()
# input_returns = df_tri.drop(['Pillar FIP', 'Pillar Equity Global'], axis=1).pct_change().dropna()
hmm = GaussianHMM(returns=input_returns)
hmm.fit(n_states=3, fit_iter=200)

hmm.trans_mat.to_excel(writer, 'Transition')
hmm.means.to_excel(writer, 'Mean')
hmm.vols.to_excel(writer, 'Vols')
(hmm.means / hmm.vols).to_excel(writer, 'Sharpe')
hmm.corrs.to_excel(writer, 'Correlations')

hmm.plot_densities()

writer.save()


# TODO add selic and GDP to regime identification
