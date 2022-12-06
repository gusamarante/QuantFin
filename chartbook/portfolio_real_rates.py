from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint
from quantfin.data import tracker_feeder, DROPBOX, SGS, tracker_uploader
from quantfin.portfolio import Performance
from quantfin.statistics import corr2cov
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

last_year = 2022
start_date = '2007-01-01'
min_vol = 0.06
rebalance_window = 3  # in months

# Excel file to save outputs
writer = pd.ExcelWriter(DROPBOX.joinpath(f'Pillar Real Rates.xlsx'))

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# =====================
# ===== Read Data =====
# =====================
# Total Return Indexes
df_tri = tracker_feeder()
df_tri = df_tri[['NTNB 0.5y', 'NTNB 1y', 'NTNB 1.5y', 'NTNB 2y', 'NTNB 3y', 'NTNB 4y',
                 'NTNB 5y', 'NTNB 6y', 'NTNB 7y', 'NTNB 8y', 'NTNB 9y', 'NTNB 10y']]

# Benchmark
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Excess Returns
df_tri = compute_eri(df_tri, df_cdi)

# Real Zero Curve
# df_raw_curve = pd.DataFrame()
# for year in tqdm(range(2003, last_year + 1), 'Reading Curves'):
#     aux = pd.read_csv(DROPBOX.joinpath(f'curves/curva_zero_ntnb_{year}.csv'))
#     aux = aux.drop(['Unnamed: 0'], axis=1)
#     df_raw_curve = pd.concat([df_raw_curve, aux])
#
# df_raw_curve = df_raw_curve.pivot('reference_date', 'du', 'yield')
#
# df_curve = 1 / ((df_raw_curve + 1) ** (df_raw_curve.columns / 252))  # Discount factors
# df_curve = np.log(df_curve)  # ln of the dicount factors
# df_curve = df_curve.interpolate(limit_area='inside', axis=1, method='index')  # linear interpolations along the lines
# df_curve = np.exp(df_curve)  # back to discounts
# df_curve = (1 / df_curve) ** (252 / df_curve.columns) - 1
# df_curve = df_curve.drop([0], axis=1)
#
# df_curve = df_curve[df_curve.index >= '2006-04-01']  # Filter the dates
# df_curve = df_curve[df_curve.columns[df_curve.columns <= 33*252]]  # Filter columns
# df_curve.index = pd.to_datetime(df_curve.index)

# ===========================
# ===== Long-run Sharpe =====
# ===========================
df_returns = df_tri / df_tri.iloc[0]
exponent = (252/np.arange(df_tri.shape[0]))
df_returns = df_returns.pow(exponent, axis=0) - 1

df_vols = df_tri.pct_change(1).expanding().std()*np.sqrt(252)
df_sharpe = df_returns / df_vols

# =======================
# ===== Backtesting =====
# =======================
dates2loop = df_tri.dropna(how='all').index
dates2loop = dates2loop[dates2loop >= start_date]
next_rebalance_date = dates2loop[0]

df_weights = pd.DataFrame(columns=df_tri.columns)
df_target_vol = pd.Series(name='Target Vol', dtype=float)
for date in tqdm(dates2loop, 'Optimizations'):

    if date >= next_rebalance_date:
        vol = df_vols.loc[date]
        mu = 0.5 * df_sharpe.loc[date] * vol
        corr = df_tri.loc[:date].resample('M').last().pct_change().corr()
        cov = corr2cov(corr, vol)

        n_assets = len(vol)
        w0 = np.ones(n_assets) * (1 / n_assets)
        bounds = Bounds(np.zeros(n_assets), np.ones(n_assets))
        cons_vol = NonlinearConstraint(fun=lambda x: np.sqrt(x @ cov @ x),
                                       lb=min_vol, ub=np.inf)
        cons_weight = LinearConstraint(A=np.ones(n_assets),
                                       lb=1, ub=1)

        def objfun(x):
            v = np.sqrt(x @ cov @ x)
            r = x @ mu
            s = r / v
            return -s

        res = minimize(objfun, w0, bounds=bounds, constraints=(cons_vol, cons_weight))

        df_target_vol.loc[date] = np.sqrt(res.x @ cov @ res.x)
        df_weights.loc[date] = res.x

        next_rebalance_date = next_rebalance_date + pd.DateOffset(months=rebalance_window)

    else:
        pass


# carry the weights trough the trimester
df_weights = df_weights.resample('D').last().fillna(method='ffill')
df_weights = df_weights.reindex(df_tri.index)
df_weights = df_weights.fillna(method='ffill')

df_weights.dropna(how='all').to_excel(writer, 'Weights')

# Excess Return Index
df_bt = df_tri.pct_change() * df_weights.dropna()
df_bt = df_bt.dropna()
df_bt = df_bt.sum(axis=1)
df_bt = (1 + df_bt).cumprod()
df_bt = 100 * df_bt / df_bt.iloc[0]
df_bt = df_bt.to_frame('Pillar Real Rate')

# performance
perf_asset = Performance(df_tri)
perf_asset.table.T.to_excel(writer, 'Asset Performance')

perf_strat = Performance(df_bt)
perf_strat.table.T.to_excel(writer, 'Strat Performance')

# Get the relevant bonds
relevant_maturities = df_weights.iloc[-1][df_weights.iloc[-1] >= 0.001].index

for bond in relevant_maturities:
    filename = bond.lower().replace(' ', '_').replace('.', '') + '.csv'
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/{filename}'), sep=';')
    aux = aux.iloc[-1]
    w = aux.loc['quantity 1'] * aux.loc['price 1'] / aux.loc['Notional']

    bond2excel = pd.Series({'bond 1': aux.loc['bond 1'],
                            'bond 2': aux.loc['bond 2'],
                            'weight 1': w,
                            'weight 2': 1 - w})

    bond2excel.to_excel(writer, bond)

writer.save()

tracker_uploader(df_bt['Pillar Real Rate'])
