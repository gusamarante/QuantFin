"""
Generates the excel file with the total return index from my XP account.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date
from bs4 import BeautifulSoup
from quantfin.data import DROPBOX
from pandas.tseries.offsets import MonthEnd

# Parameters
months = {'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
          'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12}

titles = ['Patrimônio inicial (R$)',
          'Movimentações (R$)',
          'IR pago (R$)',
          'Patrimônio final (R$)',
          'Rendimento (R$)',
          'Rentabilidade (%)',
          'Rentabilidade (% CDI)']

# Empty Dataframes
df_year = pd.DataFrame(columns=titles)
df_month = pd.DataFrame(columns=titles)
df_day = pd.DataFrame(columns=titles)

for yy in [2017, 2018, 2019, 2020, 2021, 2022]:
    with open(DROPBOX.joinpath(f'fonte da XP {yy}.txt'), encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    years2loop = soup.find_all('soma-table-row', level='year')

    for soupy in tqdm(years2loop, str(yy)):
        try:
            current_year = int(soupy.find_all('soma-table-cell')[0].string)
        except ValueError:
            continue

        cells = soupy.find_all('soma-table-cell')

        # Save Yearly Values
        df_year.loc[current_year, 'Patrimônio inicial (R$)'] = float(cells[1].string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
        df_year.loc[current_year, 'Movimentações (R$)'] = float(cells[2].string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
        df_year.loc[current_year, 'IR pago (R$)'] = float(cells[3].string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
        df_year.loc[current_year, 'Patrimônio final (R$)'] = float(cells[4].string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
        df_year.loc[current_year, 'Rendimento (R$)'] = float(cells[5].string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
        df_year.loc[current_year, 'Rentabilidade (%)'] = float(cells[6].string.replace('%', '').replace('.', '').replace(',', '.')) / 100
        df_year.loc[current_year, 'Rentabilidade (% CDI)'] = float(cells[7].string.replace('%', '').replace('.', '').replace(',', '.')) / 100

        # Save monthly values

        months2loop = soupy.find_all('soma-table-row', level='month')

        for soupm in months2loop:
            month_elements = soupm.find_all('soma-table-cell')

            find_month = list(month_elements[0].strings)
            find_month = find_month[-1].replace('\t', '').replace('\n', '')

            idx_date = pd.to_datetime(str(current_year) + '-' + str(months[find_month]) + '-1') + MonthEnd(0)

            df_month.loc[idx_date, 'Patrimônio inicial (R$)'] = float(month_elements[1].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
            df_month.loc[idx_date, 'Movimentações (R$)'] = float(month_elements[2].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
            df_month.loc[idx_date, 'IR pago (R$)'] = float(month_elements[3].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
            df_month.loc[idx_date, 'Patrimônio final (R$)'] = float(month_elements[4].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
            df_month.loc[idx_date, 'Rendimento (R$)'] = float(month_elements[5].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))

            try:
                df_month.loc[idx_date, 'Rentabilidade (%)'] = float(month_elements[6].string.replace('\t', '').replace('\n', '').replace('%', '').replace('.', '').replace(',', '.')) / 100
            except ValueError:
                df_month.loc[idx_date, 'Rentabilidade (%)'] = np.nan

            try:
                df_month.loc[idx_date, 'Rentabilidade (% CDI)'] = float(month_elements[7].string.replace('\t', '').replace('\n', '').replace('%', '').replace('.', '').replace(',', '.')) / 100
            except ValueError:
                df_month.loc[idx_date, 'Rentabilidade (% CDI)'] = np.nan

            # Save daily values
            number_days = int(len(month_elements[8:]) / 8)
            for soupd in range(1, number_days):
                day = int(month_elements[soupd * 8].string.replace('\t', '').replace('\n', ''))
                idx_day = pd.to_datetime(date(current_year, months[find_month], day))

                df_day.loc[idx_day, 'Patrimônio inicial (R$)'] = float(month_elements[soupd * 8 + 1].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
                df_day.loc[idx_day, 'Movimentações (R$)'] = float(month_elements[soupd * 8 + 2].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
                df_day.loc[idx_day, 'IR pago (R$)'] = float(month_elements[soupd * 8 + 3].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
                df_day.loc[idx_day, 'Patrimônio final (R$)'] = float(month_elements[soupd * 8 + 4].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))
                df_day.loc[idx_day, 'Rendimento (R$)'] = float(month_elements[soupd * 8 + 5].find('span').string.replace(u'\xa0', '').replace('R$', '').replace('.', '').replace(',', '.'))

                try:
                    df_day.loc[idx_day, 'Rentabilidade (%)'] = float(month_elements[soupd * 8 + 6].string.replace('\t', '').replace('\n', '').replace('%', '').replace('.', '').replace(',', '.')) / 100
                except ValueError:
                    df_day.loc[idx_day, 'Rentabilidade (%)'] = np.nan

                try:
                    df_day.loc[idx_day, 'Rentabilidade (% CDI)'] = float(month_elements[soupd * 8 + 7].string.replace('\t', '').replace('\n', '').replace('%', '').replace('.', '').replace(',', '.')) / 100
                except ValueError:
                    df_day.loc[idx_day, 'Rentabilidade (% CDI)'] = np.nan

df_year = df_year.sort_index()
df_month = df_month.sort_index()
df_day = df_day.sort_index()

df_day['Rendimento (R$)'] = df_day['Rendimento (R$)'] - df_day['Movimentações (R$)']

# Total return index
tracker = (1 + df_day['Rentabilidade (%)'].dropna()).cumprod().fillna(method='ffill')
tracker = 100 * tracker / tracker.iloc[0]
df_day['tracker'] = tracker

# Save to Excel
writer = pd.ExcelWriter(DROPBOX.joinpath('trackers/XP.xlsx'))
df_year.to_excel(writer, 'year')
df_month.to_excel(writer, 'month')
df_day.to_excel(writer, 'day')
writer.save()
