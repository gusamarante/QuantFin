from quantfin.charts import merge_pdfs
from quantfin.data import DROPBOX

chosen_assets = ['LTN Longa', 'NTNF Curta', 'NTNF Longa', 'NTNB Curta', 'NTNB Longa',
                 'BDIV11', 'IVVB', 'BBSD', 'FIND', 'GOVE', 'MATB']

pdf_writer = None

# ===== Reports =====
pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Rolling Returns.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Rolling Vol.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Rolling Sharpe.pdf'),
                        pdf_writer=pdf_writer)


pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Available Assets - Daily Clustermap.pdf'),
                        pdf_writer=pdf_writer)


# ===== Assets Data =====
pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Volatilities.pdf'),
                        pdf_writer=pdf_writer)


for asset in chosen_assets:
    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Total Return Index.pdf'),
                            pdf_writer=pdf_writer)

    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Drawdowns.pdf'),
                            pdf_writer=pdf_writer)

    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Underwater.pdf'),
                            pdf_writer=pdf_writer)

    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Correlations.pdf'),
                            pdf_writer=pdf_writer)


# ===== SAVE THE CHARTBOOK =====
with open(DROPBOX.joinpath('Portfolio Chartbook.pdf'), 'wb') as out:
    pdf_writer.write(out)
