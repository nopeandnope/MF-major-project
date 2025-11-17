"""Check table schema"""
import sqlite3

conn = sqlite3.connect('data/historical_20y.db')
cursor = conn.cursor()

# Get columns from nav_history_20y
cursor.execute("PRAGMA table_info(nav_history_20y)")
columns = cursor.fetchall()

print("nav_history_20y columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Show sample data
cursor.execute("SELECT * FROM nav_history_20y LIMIT 3")
rows = cursor.fetchall()
print("\nSample data:")
for row in rows:
    print(row)

conn.close()
