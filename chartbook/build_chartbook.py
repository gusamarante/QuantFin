from quantfin.charts import merge_pdfs
from quantfin.data import DROPBOX

chosen_assets = ['NTNB Curta', 'NTNB Longa',
                 'NTNF Curta', 'NTNF Longa',
                 'LTN Longa',
                 'SMAL',
                 'BDIV',
                 'IVVB']

pdf_writer = None

# ===== Reports =====
pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Excess Return Index.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Rolling Returns.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Rolling Vol.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/Backtests - Rolling Sharpe.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/HRP - Dendrogram.pdf'),
                        pdf_writer=pdf_writer)

pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/HRP - Correlation matrix.pdf'),
                        pdf_writer=pdf_writer)

# ===== Assets Data =====
for asset in chosen_assets:
    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Excess Return Index.pdf'),
                            pdf_writer=pdf_writer)

    try:
        pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Rolling Return.pdf'),
                                pdf_writer=pdf_writer)

        pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Rolling Vol.pdf'),
                                pdf_writer=pdf_writer)

        pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Rolling Sharpe.pdf'),
                                pdf_writer=pdf_writer)

    except FileNotFoundError:
        pass

    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Drawdowns.pdf'),
                            pdf_writer=pdf_writer)

    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Underwater.pdf'),
                            pdf_writer=pdf_writer)

    pdf_writer = merge_pdfs(DROPBOX.joinpath(f'charts/{asset} - Rolling Correlations of Assets.pdf'),
                            pdf_writer=pdf_writer)


# ===== SAVE THE CHARTBOOK =====
with open(DROPBOX.joinpath('Portfolio Chartbook.pdf'), 'wb') as out:
    pdf_writer.write(out)
