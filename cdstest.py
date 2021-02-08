from quantfin.assets import CDS

cds = CDS(200, '2002-06-22', '2007-09-20', notional=10000000)

print(cds.premium_cf)
