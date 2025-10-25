"""
db_utils.py — OCCUR-CALL AI Camera Database Utilities (Final)

Purpose
-------
- Log face events and system events to SQLite database.
- Fully self-contained; no circular imports.
- Compatible with AICameraEngine and camera_main.py.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from config import DB_PATH


# ----------------------------- Ensure DB ------------------------------ #
DB_DIR: Path = Path(DB_PATH).parent
DB_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Tables --------------------------------- #
def _ensure_face_events_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS face_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            user_id TEXT,
            image_path TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def _ensure_system_events_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()


# -------------------------- Public Logging API ------------------------ #
def log_face_event(event_type: str, user_id: str = None, image_path: str = None, confidence: float = None) -> None:
    try:
        _ensure_face_events_table()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO face_events (timestamp, event_type, user_id, image_path, confidence) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), event_type, user_id, image_path, confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Failed to log face event: {e}")


def log_system_event(event_type: str, details: str = None) -> None:
    try:
        _ensure_system_events_table()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO system_events (timestamp, event_type, details) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), event_type, details)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Failed to log system event: {e}")
