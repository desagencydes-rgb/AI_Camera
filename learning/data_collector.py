"""
data_collector.py — OCCUR-CALL Learning Data Collector (Final)

Purpose:
--------
- Collect snapshots, analysis results, and metadata from AI engine.
- Store structured data for incremental learning.
- Compatible with:
    - ai_engine.py
    - analysis modules: body_shape_analysis, clothes_detection, movement_analysis
    - DB & snapshots
"""

import os
import json
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from config import DATA_DIR

# ----------------------- Paths ----------------------- #
LEARNING_DIR = DATA_DIR / "learning_data"
SNAPSHOT_DIR = LEARNING_DIR / "snapshots"
METADATA_FILE = LEARNING_DIR / "metadata.json"

# Ensure directories exist
for d in (LEARNING_DIR, SNAPSHOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------- Analysis Modules ----------------------- #
try:
    from analysis.body_shape_analysis import analyze_body_shape
    from analysis.clothes_detection import detect_clothes
    from analysis.movement_analysis import analyze_movement
except ImportError:
    print("[WARNING] Analysis modules not found. Learning data will not include analysis results.")
    analyze_body_shape = lambda img: {}
    detect_clothes = lambda img: {}
    analyze_movement = lambda img: {}

# ----------------------- Collector ----------------------- #
def collect_learning_data(
    user_id: Optional[str],
    face_img,
    ai_metadata: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save snapshot and structured metadata for learning.

    Parameters:
    - user_id: recognized user ID or None for unknown
    - face_img: cropped face image (numpy array)
    - ai_metadata: optional AI engine output info (e.g., confidence)
    - extra_data: optional dict with any extra info (e.g., posture, activity)
    
    Returns:
    - path to saved snapshot
    """

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    user_label = str(user_id) if user_id else "unknown"
    snapshot_path = SNAPSHOT_DIR / f"{user_label}_{ts_str}.png"
    cv2.imwrite(str(snapshot_path), face_img)

    # Run analysis modules
    body_data = analyze_body_shape(face_img)
    clothes_data = detect_clothes(face_img)
    movement_data = analyze_movement(face_img)

    # Compose metadata
    metadata_entry = {
        "timestamp": ts_str,
        "user_id": user_id,
        "snapshot_path": str(snapshot_path),
        "ai_metadata": ai_metadata or {},
        "body_analysis": body_data,
        "clothes_analysis": clothes_data,
        "movement_analysis": movement_data,
        "extra_data": extra_data or {}
    }

    # Append to JSON file
    try:
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        data.append(metadata_entry)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"[LEARNING][ERROR] Failed to save metadata: {e}")

    return snapshot_path

# ----------------------- Utilities ----------------------- #
def load_all_learning_data() -> list:
    """Load all stored learning metadata."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[LEARNING][ERROR] Failed to load metadata: {e}")
            return []
    return []

def clear_learning_data():
    """Clear all learning snapshots and metadata."""
    for f in SNAPSHOT_DIR.glob("*"):
        f.unlink(missing_ok=True)
    if METADATA_FILE.exists():
        METADATA_FILE.unlink()
