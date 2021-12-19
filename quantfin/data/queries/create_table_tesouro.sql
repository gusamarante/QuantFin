CREATE TABLE raw_tesouro_direto (
    bond_type TEXT NOT NULL,
    maturity TIMESTAMP NOT NULL,
    reference_date TIMESTAMP NOT NULL,
    taxa_compra DOUBLE PRECISION,
    taxa_venda DOUBLE PRECISION,
    preco_compra DOUBLE PRECISION,
    preco_venda DOUBLE PRECISION,
    preco_base DOUBLE PRECISION,
    PRIMARY KEY (bond_type, maturity, reference_date)
);