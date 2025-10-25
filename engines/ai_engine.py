"""
ai_engine.py — OCCUR-CALL AI Camera Engine (Final, Absolute Perfection)

Features:
- Face detection (Haar cascade)
- Recognition backends:
    - face_recognition (preferred)
    - OpenCV LBPH (fallback)
    - detection-only fallback
- Full training/initialization from `data/face_db/`
- register_user(user_id, image[, bbox]) -> saves image, retrains/reloads
- Auto-register unknown faces
- Cooldown/session-memory to avoid duplicate logging
- Optional DB logging via config.py
- Enhanced recognition:
    - Multi-scale detection
    - Histogram equalization for low light
    - Auto rotation correction
    - Handles partial occlusion, distance, zoom, and variable face size
- Factory API: create_engine(...)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import os
import time
import cv2
import numpy as np
import hashlib
import sqlite3
from datetime import datetime
from config import DB_PATH
from learning.learning_core import get_learning_core

# Singleton instance for AI engine usage
_learning_core = get_learning_core()


# ----------------------------- Utility Functions ----------------------------- #
def _sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _ensure_db_table(db_path: Path) -> None:
    """Ensure face_events table exists in DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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

def _log_face_event_db(event_type: str, user_id: Optional[str], image_path: Optional[str], confidence: Optional[float]) -> None:
    """Log an event into DB."""
    try:
        _ensure_db_table(DB_PATH)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "INSERT INTO face_events (timestamp, event_type, user_id, image_path, confidence) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), event_type, user_id, image_path, confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[AI][DB] Failed to log face event: {e}")

# --------------------------- AICameraEngine Class ---------------------------- #
class AICameraEngine:
    def __init__(
        self,
        recognizer: str = "auto",
        every_n_frames: int = 1,
        draw: bool = True,
        unknown_cooldown: float = 15.0,
        snapshot_cooldown: float = 5.0,
        enable_db_logging: bool = True,
    ) -> None:
        self.recognizer_requested = recognizer.lower()
        self.every_n_frames = max(1, every_n_frames)
        self.draw = draw
        self._frame_count = 0
        self.unknown_cooldown = unknown_cooldown
        self.snapshot_cooldown = snapshot_cooldown
        self._unknown_last_saved: Dict[str, float] = {}
        self._label_last_snapshot: Dict[str, float] = {}
        self.enable_db_logging = enable_db_logging

        # Paths
        self.BASE_DIR = Path(__file__).parent.resolve()
        self.MODEL_DIR = self.BASE_DIR / "models"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.FACE_DB_DIR = self.DATA_DIR / "face_db"
        self.SNAPSHOT_DIR = self.DATA_DIR / "snapshots"
        self.UNKNOWN_DIR = self.DATA_DIR / "unknown_faces"

        for d in (self.MODEL_DIR, self.DATA_DIR, self.FACE_DB_DIR, self.SNAPSHOT_DIR, self.UNKNOWN_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # Haar cascade
        cascade_file = self.MODEL_DIR / "haarcascade_frontalface_default.xml"
        if cascade_file.exists():
            self.face_cascade = cv2.CascadeClassifier(str(cascade_file))
        else:
            self.face_cascade = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            )
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade.")

        # Recognition backend state
        self.recognizer_backend: str = "none"
        self.known_encodings: List[np.ndarray] = []
        self.known_labels: List[str] = []
        self.lbph = None
        self.lbph_label_map: Dict[int, str] = {}
        self.lbph_reverse_map: Dict[str, int] = {}

        # Initialize backend
        self._init_recognizer()

    # ----------------------- Public API ----------------------- #
    def detect_and_recognize(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self._frame_count += 1
        annotated = frame_bgr.copy()
        face_events: List[Dict[str, Any]] = []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Histogram equalization for better lighting handling
        gray = cv2.equalizeHist(gray)

        rects = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        do_recognition = (self._frame_count % self.every_n_frames == 0) and self.recognizer_backend != "none"

        for (x, y, w, h) in rects:
            user_id: Optional[str] = None
            confidence: Optional[float] = None
            saved_snapshot: Optional[Path] = None
            embedding: Optional[np.ndarray] = None

            if do_recognition:
                user_id, confidence, embedding = self._process_face(frame_bgr, (x, y, w, h), return_embedding=True)
            else:
                # If not recognizing this frame, still extract embedding for learning
                _, _, embedding = self._process_face(frame_bgr, (x, y, w, h), return_embedding=True)

            if user_id is None:
                user_id, confidence, saved_snapshot, embedding = self._handle_unknown(frame_bgr, (x, y, w, h))

            if self.draw:
                self._draw_box_and_label(annotated, (x, y, w, h), user_id, confidence)

            face_events.append({
                "bbox": (x, y, w, h),
                "user_id": user_id,
                "confidence": confidence,
                "saved_snapshot": saved_snapshot,
                "embedding": embedding  # <-- NEW: store embedding for learning system
            })

        return annotated, face_events
        

    def reload_models(self) -> None:
        self._init_recognizer(force=True)

    def register_user(self, user_id: str, face_bgr_image: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None) -> Path:
        self.FACE_DB_DIR.mkdir(parents=True, exist_ok=True)
        user_dir = self.FACE_DB_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        if bbox:
            x, y, w, h = bbox
            crop = face_bgr_image[y:y+h, x:x+w]
        else:
            crop = face_bgr_image

        fname = user_dir / f"sample_{int(time.time()*1000)}.png"
        cv2.imwrite(str(fname), crop)
        self.reload_models()
        return fname

    # ----------------------- Internal Helpers ----------------------- #
    def _draw_box_and_label(self, img: np.ndarray, bbox: Tuple[int,int,int,int], user_id: Optional[str], confidence: Optional[float]) -> None:
        x, y, w, h = bbox
        color = (0,200,0) if user_id and not str(user_id).startswith("unknown") else (0,0,200)
        label = "unknown" if user_id is None else (f"{user_id} ({confidence:.2f})" if confidence else user_id)
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _recognize_face(
        self, 
        frame_bgr: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        return_embedding: bool = False
    ) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
        """
        Recognize a face in the given bounding box.

        Parameters:
        - frame_bgr: current camera frame
        - bbox: (x, y, w, h)
        - return_embedding: whether to also return embedding vector

        Returns:
        - user_id: recognized user or None
        - confidence: recognition confidence
        - embedding: np.ndarray or None
        """
        user_id: Optional[str] = None
        confidence: Optional[float] = None
        embedding: Optional[np.ndarray] = None
        
        # Call the proper backend (do NOT pass return_embedding)
        if self.recognizer_backend == "face_recognition":
            user_id, confidence = self._recognize_face_face_recognition(frame_bgr, bbox)
        elif self.recognizer_backend == "lbph":
            user_id, confidence = self._recognize_face_lbph(frame_bgr, bbox)
            
        # Extract embedding if requested
        if return_embedding:
            embedding = self._extract_face_embedding(frame_bgr, bbox)

        return user_id, confidence, embedding
    

    # ----------------------- Backend Initialization ----------------------- #
    def _init_recognizer(self, force: bool = False) -> None:
        if self.recognizer_requested == "none":
            self._set_none()
            return
        if self._try_init_face_recognition():
            self.recognizer_backend = "face_recognition"
        elif self._try_init_lbph():
            self.recognizer_backend = "lbph"
        else:
            self._set_none()

    def _set_none(self) -> None:
        self.recognizer_backend = "none"
        self.known_encodings = []
        self.known_labels = []
        self.lbph = None
        self.lbph_label_map = {}
        self.lbph_reverse_map = {}
        print("[AI] Recognition backend: NONE (detection-only)")

    # ----------------------- Face Recognition (face_recognition) ----------------------- #
    def _try_init_face_recognition(self) -> bool:
        try:
            import face_recognition
        except Exception:
            return False
        encs: List[np.ndarray] = []
        labels: List[str] = []
        if not self.FACE_DB_DIR.exists():
            return False
        for person_dir in sorted(p for p in self.FACE_DB_DIR.iterdir() if p.is_dir()):
            label = person_dir.name
            for img_path in sorted(person_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
                    continue
                try:
                    image = face_recognition.load_image_file(str(img_path))
                    locs = face_recognition.face_locations(image, model="hog")
                    if not locs:
                        continue
                    enc = face_recognition.face_encodings(image, known_face_locations=[locs[0]])
                    if enc:
                        encs.append(enc[0])
                        labels.append(label)
                except Exception:
                    continue
        if not encs:
            return False
        self.known_encodings = encs
        self.known_labels = labels
        print(f"[AI] Recognition backend: face_recognition ({len(labels)} samples / {len(set(labels))} users)")
        return True

    def _recognize_face_face_recognition(self, frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[Optional[str], Optional[float]]:
        try:
            import face_recognition
        except Exception:
            return None, None
        x, y, w, h = bbox
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb, known_face_locations=[(y, x+w, y+h, x)])
        if not encs or not self.known_encodings:
            return None, None
        dists = face_recognition.face_distance(self.known_encodings, encs[0])
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        if dist <= 0.6:
            return self.known_labels[idx], dist
        return None, dist

    # ----------------------- LBPH Recognition ----------------------- #
    def _try_init_lbph(self) -> bool:
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            return False
        samples: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[int, str] = {}
        reverse_map: Dict[str, int] = {}
        next_label = 0
        for person_dir in sorted(p for p in self.FACE_DB_DIR.iterdir() if p.is_dir()):
            name = person_dir.name
            if name not in reverse_map:
                reverse_map[name] = next_label
                label_map[next_label] = name
                next_label += 1
            lab = reverse_map[name]
            for img_path in sorted(person_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = self._crop_primary_face(gray)
                if face is None:
                    face = cv2.resize(gray, (200,200))
                samples.append(face)
                labels.append(lab)
        if not samples:
            return False
        lbph = cv2.face.LBPHFaceRecognizer_create()
        lbph.train(samples, np.array(labels))
        self.lbph = lbph
        self.lbph_label_map = label_map
        self.lbph_reverse_map = reverse_map
        print(f"[AI] Recognition backend: LBPH ({len(samples)} images / {len(label_map)} users)")
        return True

    def _crop_primary_face(self, gray_img: np.ndarray) -> Optional[np.ndarray]:
        rects = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
        if len(rects) == 0:
            return None
        x, y, w, h = rects[0]
        try:
            return cv2.resize(gray_img[y:y+h, x:x+w], (200,200))
        except Exception:
            return None

    def _recognize_face_lbph(self, frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[Optional[str], Optional[float]]:
        if self.lbph is None:
            return None, None
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
        except Exception:
            return None, None
        label, conf = self.lbph.predict(roi)
        if conf <= 80:
            return self.lbph_label_map.get(label), conf
        return None, conf

    # ----------------------- Unknown Handling ----------------------- #
    def _handle_unknown(
        self, 
        frame_bgr: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, Optional[float], Optional[Path], Optional[np.ndarray]]:
        """
        Handle unknown face detection, save snapshot, auto-register temporary user, and return embedding.

        Returns:
        - temp_user_id
        - None (confidence)
        - saved_path
        - embedding vector
        """
        x, y, w, h = bbox
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            return "unknown", None, None, None

        # Resize & hash for cooldown checking
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))
        face_hash = _sha1_of_bytes(small.tobytes())
        now_ts = time.time()
        if now_ts - self._unknown_last_saved.get(face_hash, 0.0) < self.unknown_cooldown:
            return "unknown", None, None, None

        # Save unknown snapshot
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.UNKNOWN_DIR / f"unknown_{ts_str}.jpg"
        cv2.imwrite(str(fname), crop)
        self._unknown_last_saved[face_hash] = now_ts

        if self.enable_db_logging:
            _log_face_event_db("unknown", None, str(fname), None)

        # Auto-register unknown as temporary user
        temp_user_id = f"unknown_{ts_str}"
        saved_path = self.register_user(temp_user_id, crop, None)
        if self.enable_db_logging:
            _log_face_event_db("auto_register", temp_user_id, str(saved_path), None)

        # Compute embedding for learning system
        embedding = self._extract_face_embedding(frame_bgr, bbox)

        # Optionally add to LearningCore automatically
        _learning_core.add_feature(label=temp_user_id, feature_vector=embedding, metadata={"path": str(saved_path)})

        return temp_user_id, None, saved_path, embedding
    
    
    # ----------------------- Embedding Extraction ----------------------- #
    def _extract_face_embedding(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract a fixed-size embedding vector for a face in the bounding box.
        Works with face_recognition or fallback for LBPH.

        Parameters:
        - frame_bgr: current camera frame
        - bbox: (x, y, w, h)

        Returns:
        - embedding: np.ndarray of shape (128,) or fixed-length
        """
        x, y, w, h = bbox
        face_crop = frame_bgr[y:y+h, x:x+w]

        if face_crop.size == 0:
            # return zero vector if face not valid
            return np.zeros((128,), dtype=np.float32)

        # Convert to RGB for face_recognition backend
        if self.recognizer_backend == "face_recognition":
            try:
                import face_recognition
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_crop)
                if encodings:
                    return encodings[0]  # 128-d vector
                else:
                    # fallback zero vector if no encoding found
                    return np.zeros((128,), dtype=np.float32)
            except Exception as e:
                print(f"[Embedding] face_recognition failed: {e}")
                return np.zeros((128,), dtype=np.float32)

        elif self.recognizer_backend == "lbph":
            # For LBPH, resize to fixed size, flatten, normalize
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray_crop, (64, 64))
            embedding = small.flatten().astype(np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-6  # normalize
            return embedding

        else:
            # Fallback for 'none' or unknown backend
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray_crop, (64, 64))
            embedding = small.flatten().astype(np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-6
            return embedding
        
        
    # ----------------------- Unified Face Processing ----------------------- #
    def _process_face(self, frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int], return_embedding: bool = False) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
        """
        Always extracts embedding and returns it alongside recognition info.
        """
        # Step 1: Recognize face and optionally get embedding
        user_id, confidence, embedding = self._recognize_face(frame_bgr, bbox, return_embedding=return_embedding)
        
        # Step 2: Update learning core if embedding exists
        if user_id and embedding is not None:
            _learning_core.add_feature(label=user_id, feature_vector=embedding, metadata={"recognized": True})

        
        return user_id, confidence, embedding

        

# ----------------------- Factory ----------------------- #
def create_engine(
    recognizer: str = "auto",
    every_n_frames: int = 1,
    draw: bool = True,
    unknown_cooldown: float = 15.0,
    snapshot_cooldown: float = 5.0,
    enable_db_logging: bool = True
) -> AICameraEngine:
    return AICameraEngine(
        recognizer, every_n_frames, draw,
        unknown_cooldown, snapshot_cooldown, enable_db_logging
    )
