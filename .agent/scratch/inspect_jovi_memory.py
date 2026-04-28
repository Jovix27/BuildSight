import sys
import os
import sqlite3
import json

# Set encoding to UTF-8 for windows terminal
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

db_path = r'e:\Company\Green Build AI\R&D\Jovi Claw\jovi_memory.db'

def check_db():
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch symphony tasks
        print("\n--- symphony_tasks ---")
        cursor.execute("SELECT * FROM symphony_tasks")
        tasks = cursor.fetchall()
        for task in tasks:
            d = dict(task)
            print(json.dumps(d, indent=2, ensure_ascii=False))

        # Fetch symphony events
        print("\n--- symphony_events (last 10) ---")
        cursor.execute("SELECT * FROM symphony_events ORDER BY created_at DESC LIMIT 10")
        events = cursor.fetchall()
        for event in events:
            d = dict(event)
            print(json.dumps(d, indent=2, ensure_ascii=False))

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
