"""
Creates the empty database with the desired structure.
"""

import sqlite3

conn = sqlite3.connect('database_finquant.db')
cursor = conn.cursor()

# Create Table for Tesouro Direto
query = open('queries/create_tables.sql').read()
cursor.execute(query)
