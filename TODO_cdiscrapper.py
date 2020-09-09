"""
- check the last available date on the database
- scrape values since
- upload values to database
"""

import pandas as pd

start_date = pd.to_datetime('2018-01-02')
end_date = pd.to_datetime('2018-08-31')

dates = pd.date_range(start_date, end_date, freq='D')

for d in dates:

    try:
        media = pd.read_csv(f'ftp://ftp.cetip.com.br/MediaCDI/{d.strftime("%Y%m%d")}.txt',
                            header=None).iloc[0, 0] / 100

        volume = pd.read_csv(f'ftp://ftp.cetip.com.br/VolumeCDI/{d.strftime("%Y%m%d")}.txt',
                             header=None).iloc[0, 0]

    except:
        continue

    # save media and volume here
