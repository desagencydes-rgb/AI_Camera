"""
camera_main.py — OCCUR-CALL AI Camera Main (Final v4)

Purpose
-------
- Capture frames from camera.
- Detect and recognize faces in real-time using AICameraEngine.
- Auto-save unknown faces.
- Auto-log recognized & unknown faces into database.
- Overlay boxes: green for recognized, red for unknown.
- Optional enhancements via ai_engine_enhancer.py
- Console output with structured event info.
"""

import cv2
import os
import sys
from datetime import datetime
from engines.ai_engine import AICameraEngine
from Utils.db_utils import log_face_event
from engines.ai_engine_enhancer import enhance_engine
from config import DB_PATH  # Make sure DB_PATH points to your face_events.db
from learning.learning_hooks import trigger_hooks


# ========================
# Configuration
# ========================
ENABLE_DB_LOGGING = True
USE_ENHANCER = True  # Set to True to enable ai_engine_enhancer features

SNAPSHOT_DIR = r"C:\AI_Camera\data\snapshots"
UNKNOWN_DIR = r"C:\AI_Camera\data\unknown_faces"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ========================
# Initialize AI Engine
# ========================
try:
    ai = AICameraEngine(recognizer="auto", enable_db_logging=ENABLE_DB_LOGGING)
except Exception as e:
    print(f"[FATAL] Failed to initialize AI Engine: {e}")
    sys.exit(1)

# Apply optional enhancer
if USE_ENHANCER:
    ai = enhance_engine(ai)

# Set directories in AI engine (if not already)
ai.snapshot_dir = SNAPSHOT_DIR
ai.unknown_dir = UNKNOWN_DIR

print("=" * 60)
print(f"[AI] Recognition backend: {ai.recognizer_backend.upper()}")
print(f"[AI] Loaded {len(ai.known_labels)} samples "
      f"across {len(set(ai.known_labels))} unique users")
print("[AI] System ready. Starting camera...")
print("=" * 60)

# ========================
# Helper Functions
# ========================
def save_unknown_snapshot(face_img) -> str:
    """Save unknown face snapshot and return path."""
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(UNKNOWN_DIR, f"unknown_{ts_str}.png")
    cv2.imwrite(path, face_img)
    return path

# ========================
# Camera Setup
# ========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[FATAL] Could not access camera. Check connection.")
    sys.exit(1)

# ========================
# Camera Loop
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    try:
        annotated_frame, face_events = ai.detect_and_recognize(frame)
    except Exception as e:
        print(f"[AI ERROR] Processing frame failed: {e}")
        continue

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for face in face_events:
        x, y, w, h = face["bbox"]
        user_id = face["user_id"]
        confidence = face.get("confidence")
        saved_snapshot = face.get("saved_snapshot")

        # Determine box color and label
        if user_id is None or str(user_id).startswith("unknown"):
            color = (0, 0, 200)  # Red for unknown
            label = "unknown"
            if saved_snapshot is None:
                face_crop = frame[y:y+h, x:x+w]
                saved_snapshot = save_unknown_snapshot(face_crop)
            if ENABLE_DB_LOGGING:
                log_face_event(event_type="unknown", user_id=None,
                               image_path=saved_snapshot, confidence=None)
        else:
            color = (0, 200, 0)  # Green for recognized
            label = f"{user_id} ({confidence:.2f})" if confidence else user_id
            if ENABLE_DB_LOGGING:
                log_face_event(event_type="recognized", user_id=user_id,
                               image_path=None, confidence=confidence)

        # Draw box and label
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated_frame, label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Console output
        if user_id is None or str(user_id).startswith("unknown"):
            print(f"[{ts}] Unknown face detected | Snapshot: {saved_snapshot}")
        else:
            conf_txt = f" ({confidence:.2f})" if confidence else ""
            print(f"[{ts}] Recognized: {user_id}{conf_txt}")
            
            
        # Trigger all registered hooks for analysis / additional engines
        event_data = {
            "frame": frame,          # current camera frame
            "faces": face_events,    # list of detected faces with user_id, bbox, etc.
            "timestamp": datetime.now()
        }
        trigger_hooks(event_data, async_mode=True)
        

    # Display frame
    cv2.imshow("OCCUR-CALL Camera", annotated_frame)

    # Exit keys: q or ESC
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        print("[AI] Exit requested by user.")
        break
    elif key == ord("r"):  # r to reload AI engine models
        print("[AI] Reloading face models...")
        ai.reload_models()

# ========================
# Cleanup
# ========================
cap.release()
cv2.destroyAllWindows()
print("[AI] Camera stopped cleanly.")
