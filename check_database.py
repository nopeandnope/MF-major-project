"""Check what tables exist in the database"""
import sqlite3

db_path = 'data/historical_20y.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if tables:
        print(f"Database has {len(tables)} tables:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} rows")
    else:
        print("Database exists but has NO TABLES (empty database)")
        print("\nYou need to run data collection first!")

    conn.close()

except Exception as e:
    print(f"Error: {e}")
