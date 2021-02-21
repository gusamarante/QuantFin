import matplotlib.pyplot as plt
from quantfin.simulation import Diffusion

diff = Diffusion(T=1, n=100, k=1000)

diff.brownian_motion.plot(legend=None)
plt.show()

print(diff.brownian_motion.iloc[-1].mean())
print(diff.brownian_motion.iloc[-1].std())

plt.hist(diff.brownian_motion.iloc[-1], bins=100)
plt.show()
