from config import OCCUR_DB
DB_PATH = OCCUR_DB

import sqlite3

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        role TEXT,
        face_encoding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # attendance logs
    c.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        status TEXT CHECK(status IN ('recognized', 'manual', 'corrected')),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        snapshot_path TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # unknown faces
    c.execute("""
    CREATE TABLE IF NOT EXISTS unknown_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_path TEXT,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP,
        seen_count INTEGER DEFAULT 1
    )
    """)

    # optional migration from old table
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='unknown'")
        if c.fetchone():
            print("[DB] Migrating old 'unknown' table data...")
            c.execute("INSERT INTO unknown_faces (snapshot_path, first_seen, last_seen, seen_count) "
                      "SELECT snapshot_path, first_seen, last_seen, seen_count FROM unknown")
            c.execute("DROP TABLE unknown")
            print("[DB] Migration complete. Old 'unknown' table removed.")
    except Exception as e:
        print(f"[DB] Migration check skipped: {e}")

    conn.commit()
    conn.close()
    print("[DB] Database initialized and updated successfully.")


if __name__ == "__main__":
    initialize_db()
