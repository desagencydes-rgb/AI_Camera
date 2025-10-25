"""
incremental_trainer.py — OCCUR-CALL AI Engine Incremental Trainer

Purpose
-------
- Incrementally train/update AI Engine models from collected learning data.
- Non-blocking: runs in a separate thread.
- Avoids reprocessing already trained data.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import threading
import time
import json
import os
import cv2

from engines.ai_engine import AICameraEngine  # main AI engine
from config import DATA_DIR

# Paths
LEARNING_DIR = Path(DATA_DIR) / "learning_data"
EVENTS_JSON = LEARNING_DIR / "events.json"
PROCESSED_JSON = LEARNING_DIR / "processed_events.json"

class IncrementalTrainer:
    """
    Incrementally updates AI Engine based on learning data.
    """

    def __init__(self, ai_engine: AICameraEngine, interval: float = 15.0):
        """
        Parameters:
        - ai_engine: instance of AICameraEngine
        - interval: seconds between incremental training runs
        """
        self.ai = ai_engine
        self.interval = interval
        self._stop_flag = False
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()

    # -----------------------------
    # Load events
    # -----------------------------
    def _load_events(self) -> List[dict]:
        if not EVENTS_JSON.exists():
            return []
        try:
            with EVENTS_JSON.open("r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception:
            events = []
        return events

    def _load_processed_ids(self) -> set:
        if not PROCESSED_JSON.exists():
            return set()
        try:
            with PROCESSED_JSON.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data)
        except Exception:
            return set()

    def _save_processed_ids(self, processed_ids: set):
        try:
            with PROCESSED_JSON.open("w", encoding="utf-8") as f:
                json.dump(list(processed_ids), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TRAINER] Failed to save processed IDs: {e}")

    # -----------------------------
    # Training Loop
    # -----------------------------
    def _train_loop(self):
        processed_ids = self._load_processed_ids()
        while not self._stop_flag:
            events = self._load_events()
            new_processed = set()

            for idx, event in enumerate(events):
                if idx in processed_ids:
                    continue

                if event.get("type") == "face_event" and event.get("saved_snapshot"):
                    user_id = event.get("user_id")
                    path = event.get("saved_snapshot")
                    if path and Path(path).exists() and user_id:
                        try:
                            # Register user in AI engine incrementally
                            img = cv2.imread(path)
                            if img is not None:
                                self.ai.register_user(user_id, img)
                                print(f"[TRAINER] Incrementally trained user '{user_id}' from {path}")
                        except Exception as e:
                            print(f"[TRAINER] Failed to train user {user_id}: {e}")

                # Mark this event as processed
                new_processed.add(idx)

            # Update processed IDs
            processed_ids.update(new_processed)
            self._save_processed_ids(processed_ids)

            # Sleep until next incremental training cycle
            time.sleep(self.interval)

    def stop(self):
        """Stop incremental trainer cleanly."""
        self._stop_flag = True
        self._thread.join(timeout=3.0)
