"""
Creates the empty database with the desired structure.
"""

from quantfin.data import grab_connection

conn = grab_connection()
cursor = conn.cursor()

# Create Table for Tesouro Direto
query = open('queries/create_tables.sql').read()
cursor.execute(query)
