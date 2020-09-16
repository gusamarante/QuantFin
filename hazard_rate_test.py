import pandas as pd
from quantfin.assets import HazardRateTermStructure

hr = HazardRateTermStructure()
hr.add_hazard(360, 0.000010)
hr.add_hazard(720, 0.000015)

print(hr.survival_prob(0, 540))
