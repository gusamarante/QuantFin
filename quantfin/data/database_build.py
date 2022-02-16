"""
Creates the empty database with the desired structure.
"""

from quantfin.data import grab_connection

conn = grab_connection()
cursor = conn.cursor()

# Create table for the trackers
query = open('queries/create_table_trackers.sql').read()
cursor.execute(query)

# Close the cursor
cursor.close()
