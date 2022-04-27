from hmmlearn import hmm
import pandas as pd
import numpy as np


class GaussianHMM(object):

    def __init__(self, df_eri, n_states=None):
        # TODO Documentation - format is focused in finance
        # TODO result tables - Prob of persistance, average duration, relative frequency
        # TODO plot methods - return, std and sharpe by state, eri with states in the background, matrix transitions, distributions by state

        self.n_var = df_eri.shape[1]

        if n_states is None:
            pass  # TODO if state is none, compute several and grab the best
        else:
            self.n_states = n_states

        df_ret = df_eri.pct_change(1).dropna()
        model = hmm.GaussianHMM(n_components=self.n_states,
                                covariance_type='full',
                                n_iter=100)
        model.fit(df_ret)
        pred_state = pd.Series(data=model.predict(df_ret),
                               index=df_ret.index,
                               name='Predicted State')
        df_ret['State'] = pred_state + 1

        self.state_means = (df_ret.groupby('State').mean() + 1) ** 252 - 1
        self.state_stds = df_ret.groupby('State').std() * np.sqrt(252)
        self.state_sharpe = self.state_means / self.state_stds

        self.trans_mat = pd.DataFrame(data=model.transmat_,
                                      index=[f'From State {s + 1}' for s in range(self.n_states)],
                                      columns=[f'To State {s + 1}' for s in range(self.n_states)])

        self.average_duration = pd.Series(index=[f'State {s + 1}' for s in range(self.n_states)],
                                          data=1 / np.diag(self.trans_mat.values),
                                          name='Average Duration')
