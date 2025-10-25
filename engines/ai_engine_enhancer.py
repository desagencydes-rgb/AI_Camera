# ai_engine_enhancer.py
"""
OCCUR-CALL AI Camera Engine Enhancer

Purpose:
- Optional enhancements for AICameraEngine.
- Improves recognition under challenging conditions:
  lighting changes, rotation, partial occlusion, zoom, far faces.
- Adds threaded recognition for performance.
- Enhances unknown-face handling.
- Does NOT modify the core engine.
"""

from typing import Tuple, Optional, Callable
import cv2
import numpy as np
import threading
from datetime import datetime
import time

# ---------------------------- Image Enhancements --------------------------- #
def normalize_lighting(face_gray: np.ndarray) -> np.ndarray:
    """Equalize histogram to improve lighting conditions."""
    return cv2.equalizeHist(face_gray)

def rotate_face(face_img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center."""
    h, w = face_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(face_img, M, (w, h))

def scale_face(face_img: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale image up/down to handle zoom/far faces."""
    h, w = face_img.shape[:2]
    new_size = (max(1, int(w*scale_factor)), max(1, int(h*scale_factor)))
    return cv2.resize(face_img, new_size)

def enhance_face_crop(face_img: np.ndarray) -> np.ndarray:
    """Apply all basic enhancements: lighting and minor scaling."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    norm = normalize_lighting(gray)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

# ---------------------------- Threaded Recognition ------------------------- #
def run_recognition_threaded(engine, frame, bbox, callback: Callable):
    """Run recognition in a separate thread and call callback(user_id, confidence)."""
    def task():
        user_id, confidence = engine._recognize_face(frame, bbox)
        callback(user_id, confidence)
    t = threading.Thread(target=task)
    t.daemon = True
    t.start()
    return t

# ---------------------------- Enhanced Unknown Handling -------------------- #
def enhanced_handle_unknown(engine, frame, bbox) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Enhances unknown-face handling:
    - Lighting normalization
    - Multiple rotations
    - Small scaling
    - Falls back to original _handle_unknown
    """
    x, y, w, h = bbox
    h_frame, w_frame = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_frame, x+w), min(h_frame, y+h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return ("unknown", None, None)

    # Apply lighting normalization
    enhanced_crop = enhance_face_crop(crop)

    # Try multiple rotations and scales
    angles = [0, -15, 15]
    scales = [1.0, 1.2, 0.8]

    best_user = None
    best_conf = None

    for scale in scales:
        scaled = scale_face(enhanced_crop, scale)
        for angle in angles:
            rotated = rotate_face(scaled, angle)
            user_id, conf = engine._recognize_face(rotated, (0, 0, rotated.shape[1], rotated.shape[0]))
            if user_id is not None:
                best_user = user_id
                best_conf = conf
                break
        if best_user is not None:
            break

    # Fallback to original unknown handling
    if best_user is None:
        return engine._handle_unknown(frame, bbox)

    return (best_user, best_conf, None)

# ---------------------------- Session Memory Enhancer ---------------------- #
class SessionMemory:
    """
    Keeps track of recently recognized faces to avoid duplicate logging.
    Can be integrated with AICameraEngine._unknown_last_saved
    """
    def __init__(self, cooldown_seconds: float = 15.0):
        self.cooldown = cooldown_seconds
        self.last_seen = {}  # face_hash -> timestamp

    def should_log(self, face_hash: str) -> bool:
        now = time.time()
        last = self.last_seen.get(face_hash, 0)
        if now - last >= self.cooldown:
            self.last_seen[face_hash] = now
            return True
        return False

# ---------------------------- Helper Logging ------------------------------- #
def log_enhanced_event(engine, user_id: str, event_type: str, image_path: Optional[str] = None, confidence: Optional[float] = None):
    """
    Log face events using engine's DB connection if enabled.
    """
    if getattr(engine, 'enable_db_logging', False):
        try:
            engine._log_face_event_db(event_type, user_id, image_path, confidence)
        except Exception as e:
            print(f"[ENHANCER][DB] Failed to log event: {e}")

# ---------------------------- Factory -------------------------------------- #
def enhance_engine(engine):
    """
    Attach enhancer functions to the engine dynamically:
    - Replace _handle_unknown with enhanced version
    - Add threaded recognition helper
    - Add session memory
    """
    engine.original_handle_unknown = engine._handle_unknown
    engine._handle_unknown = lambda frame, bbox: enhanced_handle_unknown(engine, frame, bbox)
    engine.run_recognition_threaded = lambda frame, bbox, callback: run_recognition_threaded(engine, frame, bbox, callback)
    engine.session_memory = SessionMemory()
    return engine
