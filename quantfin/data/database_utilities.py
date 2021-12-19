import sqlite3
import pandas as pd


def grab_connection():
    # TODO Documentation
    # TODO grab path to project
    conn = sqlite3.connect('/Users/gustavoamarante/PycharmProjects/QuantFin/quantfin/data/database_finquant.db')
    return conn


def tracker_delete(names, conn=None):
    # TODO Documentation

    if conn is None:
        conn = grab_connection()

    if isinstance(names, list):
        name_list = "('" + "', '".join(names) + "')"
        query = f"delete from 'trackers' where variable in {name_list}"
    elif isinstance(names, str):
        query = f"delete from 'trackers' where variable = '{names}'"
    else:
        raise ValueError("'names' format is not accepted")

    cursor = conn.cursor()
    cursor.execute(str(query))
    conn.commit()
    cursor.close()


def tracker_uploader(data, conn=None):
    # TODO Documentation (data must have a DateTime index)

    # Makes sure that the Index is DateTime
    data.index = pd.to_datetime(list(data.index))

    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    # Drop the old trackers
    tracker_names = list(data.columns)
    tracker_delete(tracker_names, conn)

    # Put data in the "melted" format
    data = data.reset_index()
    data = data.melt('index')

    # upload the new trackers
    data.to_sql('trackers', con=conn, index=False)
