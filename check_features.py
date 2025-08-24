#!/usr/bin/env python3
import sqlite3
import pandas as pd
import os

# Check what database files exist
print("Current directory:", os.getcwd())
print("Files in current dir:", [f for f in os.listdir('.') if f.endswith('.db')])

# Try the default database path
db_path = 'arbitrage.db'
if not os.path.exists(db_path):
    print(f"Database {db_path} does not exist")
    exit()

# Connect to database
conn = sqlite3.connect(db_path)

# Check tables
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = [row[0] for row in cursor.fetchall()]

# Find feature tables
feature_tables = [t for t in tables if 'feature' in t.lower()]
print("Feature tables:", feature_tables)

# Check signal tables
signal_tables = [t for t in tables if 'signal' in t.lower()]
print("Signal tables:", signal_tables)

# Show sample features if any exist
for table in feature_tables:
    try:
        df = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 5', conn)
        print(f"\nTable: {table}")
        print(df)
    except Exception as e:
        print(f"Error reading {table}: {e}")

# Show sample signals if any exist  
for table in signal_tables:
    try:
        df = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 5', conn)
        print(f"\nTable: {table}")
        print(df)
    except Exception as e:
        print(f"Error reading {table}: {e}")

conn.close()
