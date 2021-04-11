import sqlite3


def grab_connection():
    # TODO grab path to project
    conn = sqlite3.connect('/Users/gustavoamarante/PycharmProjects/QuantFin/quantfin/data/database_finquant.db')
    return conn

# TODO tracker feeder
