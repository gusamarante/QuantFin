import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from hmmlearn import hmm
from tqdm import tqdm
import pandas as pd
import numpy as np


class GaussianHMM(object):

    n_states = None

    def __init__(self, returns):
        # TODO Documentation - format is focused in finance
        # TODO result tables - Prob of persistance, average duration, relative frequency
        # TODO plot methods - return, std and sharpe by state, eri with states in the background, matrix transitions, distributions by state + normal mixture of the states
        # TODO notebook with functionalities

        self.returns = returns
        self.n_var = returns.shape[1]


        #
        #     self.n_states = self.state_selection.diff().diff().shift(-1).idxmax()
        #
        # else:
        #     self.state_selection = None
        #     self.n_states = n_states
        #
        #
        # # TODO this needs to run billions of times.
        # # TODO understand the priors
        # model = hmm.GaussianHMM(n_components=self.n_states,
        #                         covariance_type='full',
        #                         n_iter=1000)
        # model.fit(df_ret)
        # self.pred_state = pd.Series(data=model.predict(df_ret),
        #                             index=df_ret.index,
        #                             name='Predicted State')
        # df_ret['State'] = self.pred_state + 1
        #
        # self.state_freq = (self.pred_state.value_counts() / self.pred_state.count()).sort_index().rename('Frequency')
        # self.state_means = (df_ret.groupby('State').mean() + 1) ** 252 - 1
        # self.state_stds = df_ret.groupby('State').std() * np.sqrt(252)
        # self.state_sharpe = self.state_means / self.state_stds
        #
        # self.trans_mat = pd.DataFrame(data=model.transmat_,
        #                               index=[f'From State {s + 1}' for s in range(self.n_states)],
        #                               columns=[f'To State {s + 1}' for s in range(self.n_states)])
        #
        # self.average_duration = pd.Series(index=[f'State {s + 1}' for s in range(self.n_states)],
        #                                   data=1 / np.diag(self.trans_mat.values),
        #                                   name='Average Duration')

    def select_order(self, max_state_number=10, select_iter=10, show_chart=False):
        self.state_selection = pd.Series(name='HMM Score')

        for ns in tqdm(range(1, max_state_number + 1), 'Computing scores for a different number of states'):
            max_score = - np.inf

            for i in range(select_iter):
                model = hmm.GaussianHMM(n_components=ns,
                                        covariance_type='full',
                                        n_iter=1000)
                model.fit(self.returns)
                new_score = model.score(self.returns)
                if new_score > max_score:
                    max_score = new_score

            self.state_selection.loc[ns] = max_score

        if show_chart:
            most_concave = self.state_selection.diff().diff().idxmin()
            ax = self.state_selection.plot()
            loc = plticker.MultipleLocator(base=1)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Number of States')
            ax.set_ylabel('Log-likelihood')
            ax.axvline(most_concave, color='red')
            ymin, ymax = ax.get_ylim()
            height = ymin + (ymax - ymin) * 0.05
            ax.text(most_concave + 0.2, height, '"Elbow" Rule - Most concave point',
                    size=8, horizontalalignment='left', backgroundcolor='red',
                    color='red',
                    bbox=dict(alpha=.8, facecolor='w'))
            plt.tight_layout()
            plt.show()
