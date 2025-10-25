"""
config.py — OCCUR-CALL AI Camera / System Configuration
"""

from pathlib import Path

# ---------------------------
# Base directories
# ---------------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
FACE_DB_DIR = DATA_DIR / "face_db"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
UNKNOWN_DIR = DATA_DIR / "unknown_faces"
MODEL_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "Update_and_Backup"

# ---------------------------
# Database paths (absolute)
# ---------------------------
# Main OCCUR-CALL system DB
OCCUR_DB = Path(r"C:\OCCUR-CALL\Backend\instance\occur-call.db")

# AI Camera face events DB
DB_PATH = DB_DIR / "face_events.db"

# ---------------------------
# Optional: Haar cascade override
# ---------------------------
# If you want to force a custom Haar cascade, set path here:
# CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"

# ---------------------------
# Ensure directories exist
# ---------------------------
for d in (DATA_DIR, FACE_DB_DIR, SNAPSHOT_DIR, UNKNOWN_DIR, MODEL_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)
