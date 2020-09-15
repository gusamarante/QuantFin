from quantfin.finmath import HazardRateTermStructure
import pandas as pd

hr = HazardRateTermStructure()
hr.add_hazard(360, 0.000010)
hr.add_hazard(720, 0.000015)

print(hr.survival_prob(0, 360))
