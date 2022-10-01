"""
Pricing an up-and-out call with simulation and comparing with the analytical formula
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters
S0 = 100
E = 100
Sb = 130
r = 0.04
q = 0
sigma = 0.2
T = 1

n_simul = 10000000


# ===== Analytical price =====
d1 = (np.log(S0 / E) + (r - q + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d2 = (np.log(S0 / E) + (r - q - 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d3 = (np.log(S0 / Sb) + (r - q + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d4 = (np.log(S0 / Sb) + (r - q - 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d5 = (np.log(S0 / Sb) - (r - q - 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d6 = (np.log(S0 / Sb) - (r - q + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d7 = (np.log(S0 * E / (Sb ** 2)) - (r - q - 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
d8 = (np.log(S0 * E / (Sb ** 2)) - (r - q + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
a = (Sb / S0) ** (2 * (r - q) / (sigma ** 2) - 1)
b = (Sb / S0) ** (2 * (r - q) / (sigma ** 2) + 1)

nd1 = norm.cdf(d1)
nd2 = norm.cdf(d2)
nd3 = norm.cdf(d3)
nd4 = norm.cdf(d4)
nd5 = norm.cdf(d5)
nd6 = norm.cdf(d6)
nd7 = norm.cdf(d7)
nd8 = norm.cdf(d8)

vanilla_price = - E * np.exp(-r * T) * nd2 + S0 * np.exp(-q * T) * nd1

barrier_price = S0 * np.exp(-q * T) * (nd1 - nd3 - b * (nd6 - nd8)) \
              - E * np.exp(-r * T) * (nd2 - nd4 - a * (nd5 - nd7))

# ===== Simulated Pricing =====
delta_t = 1 / (252 * T)
returns = np.random.normal((r - 0.5*sigma**2)*delta_t, sigma*np.sqrt(delta_t), (252*T, n_simul))
returns = np.cumsum(returns, axis=0)
simulated_trajectories = S0 * np.exp(returns)

vanilla_simul = np.mean(np.maximum(simulated_trajectories[-1] - E, 0)) * np.exp(-r*T)

live_options = simulated_trajectories.max(axis=0) < Sb
barrier_simul = np.mean(live_options * np.maximum(simulated_trajectories[-1] - E, 0)) * np.exp(-r*T)

# Print Results
print('vanilla analytical price', vanilla_price)
print('vanilla simulated price', vanilla_simul)
print('barrier analytical price', barrier_price)
print('barrier simulated price', barrier_simul)
