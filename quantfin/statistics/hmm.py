from quantfin.statistics import cov2corr
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from hmmlearn import hmm
from tqdm import tqdm
import pandas as pd
import numpy as np


class GaussianHMM(object):
    # TODO Documentation
    #      - format is focused in finance
    #      - kind of a wrapper for DFs
    #      - our interests
    #      - instability of the estimator

    # TODO Stuff to check
    #      - understand the options for prior parameters

    # TODO plot methods
    #      - DF with states in the background
    #      - networkx de transição de states
    #      - distributions by state + normal mixture of the states

    # TODO before showing
    #      - Examples with simulated data, to show consistency
    #      - Example with real data, to see it in action, with all the examples from my file

    predicted_state = None
    state_selection = None
    stationary_dist = None
    avg_duration = None
    state_probs = None
    state_freq = None
    trans_mat = None
    n_states = None
    covars = None
    score = None
    corrs = None
    means = None
    vols = None

    def __init__(self, returns):
        self.returns = returns
        self.n_var = returns.shape[1]

    def select_order(self, max_state_number=8, select_iter=10, show_chart=False):
        self.state_selection = pd.Series(name='HMM Score')

        for ns in tqdm(range(1, max_state_number + 1), 'Computing scores for a different number of states'):
            max_score = - np.inf

            for _ in range(select_iter):
                model = hmm.GaussianHMM(n_components=ns,
                                        covariance_type='full',
                                        n_iter=1000)
                model.fit(self.returns)
                new_score = model.score(self.returns)
                if new_score > max_score:
                    max_score = new_score

            self.state_selection.loc[ns] = max_score

        most_concave = self.state_selection.diff().diff().idxmin()

        if show_chart:
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

        return most_concave

    def fit(self, fit_iter=100, n_states=None, max_state_number=8, select_iter=10):

        if n_states is None:
            self.n_states = self.select_order(max_state_number=max_state_number,
                                              select_iter=select_iter,
                                              show_chart=False)
        else:
            self.n_states = n_states

        # Estimate the model several times, due to instability, and grab the one with the highest score.
        model_dict = dict()
        for _ in tqdm(range(fit_iter), 'Estimating HMM'):
            model = hmm.GaussianHMM(n_components=self.n_states,
                                    covariance_type='full',
                                    n_iter=1000)
            model.fit(self.returns)
            model_dict[model.score(self.returns)] = model

        chosen_model = model_dict[max(model_dict.keys())]
        sort_order = np.flip(np.argsort(np.diag(chosen_model.transmat_)))

        # Build the sorted model
        sorted_model = hmm.GaussianHMM(n_components=self.n_states,
                                       covariance_type='full')

        sorted_model.startprob_ = chosen_model.startprob_[sort_order]
        sorted_model.transmat_ = pd.DataFrame(chosen_model.transmat_).loc[sort_order, sort_order].values
        sorted_model.means_ = chosen_model.means_[sort_order, :]
        sorted_model.covars_ = chosen_model.covars_[sort_order, :, :]

        self.score = sorted_model.score(self.returns)

        self.trans_mat = pd.DataFrame(data=sorted_model.transmat_,
                                      index=[f'From State {s + 1}' for s in range(self.n_states)],
                                      columns=[f'To State {s + 1}' for s in range(self.n_states)])

        self.avg_duration = pd.Series(data=1 / (1 - np.diag(sorted_model.transmat_)),
                                      index=[f'State {s + 1}' for s in range(self.n_states)],
                                      name='Average Duration')

        self.stationary_dist = pd.Series(data=sorted_model.get_stationary_distribution(),
                                         index=[f'State {s + 1}' for s in range(self.n_states)],
                                         name='Stationary Distribution of States')

        self.means = pd.DataFrame(data=sorted_model.means_,
                                  index=[f'State {s + 1}' for s in range(self.n_states)],
                                  columns=self.returns.columns)

        vol_data = [list(np.sqrt(np.diag(sorted_model.covars_[ss]))) for ss in range(self.n_states)]
        self.vols = pd.DataFrame(data=vol_data, columns=self.returns.columns,
                                 index=[f'State {s + 1}' for s in range(self.n_states)])

        idx = pd.MultiIndex.from_product([[f'State {s + 1}' for s in range(self.n_states)],
                                          self.returns.columns])
        self.covars = pd.DataFrame(index=idx, columns=self.returns.columns,
                                   data=sorted_model.covars_.reshape(-1, self.n_var))

        corr_data = [cov2corr(sorted_model.covars_[ss])[0] for ss in range(self.n_states)]
        self.corrs = pd.DataFrame(index=idx, columns=self.returns.columns,
                                  data=np.concatenate(corr_data))

        self.predicted_state = pd.Series(data=sorted_model.predict(self.returns) + 1,
                                         index=self.returns.index,
                                         name='Predicted State')

        freq_data = ('State ' + self.predicted_state.astype(str)).value_counts() / self.predicted_state.count()
        self.state_freq = pd.Series(data=freq_data,
                                    index=[f'State {s + 1}' for s in range(self.n_states)],
                                    name='State Frequency')

        self.state_probs = pd.DataFrame(data=sorted_model.predict_proba(self.returns),
                                        index=self.returns.index,
                                        columns=[f'State {s + 1}' for s in range(self.n_states)])
