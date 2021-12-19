CREATE TABLE trackers (
    'index' TIMESTAMP NOT NULL,
    'variable' TEXT NOT NULL,
    'value' DOUBLE PRECISION NOT NULL,
    PRIMARY KEY ('index', variable)
);

