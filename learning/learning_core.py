"""
learning_core.py — OCCUR-CALL Learning Core

Purpose
-------
- Manage storage, retrieval, and update of embeddings/features.
- Support various learning data: faces, body shapes, objects.
- Provide quick similarity queries.
- Fully integratable with learning_hooks for external engines/analysis.
"""

import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ----------------------- Paths & Storage ----------------------- #
BASE_DIR = Path(__file__).parent.parent.resolve()
LEARNING_DB_DIR = BASE_DIR / "data" / "learning_db"
LEARNING_DB_DIR.mkdir(parents=True, exist_ok=True)
LEARNING_DB_FILE = LEARNING_DB_DIR / "learning_data.pkl"

# ----------------------- Learning Core ------------------------ #
class LearningCore:
    """
    Core manager for storing, updating, and querying learned features.
    """

    def __init__(self):
        self._data_lock = threading.Lock()
        self._features: List[np.ndarray] = []
        self._labels: List[str] = []
        self._meta: Dict[str, Dict[str, Any]] = {}  # Optional metadata per label
        self._nn_index: Optional[NearestNeighbors] = None

        # Load existing data if available
        self._load_db()

    # ------------------- Public API ---------------------------- #
    def add_feature(self, label: str, feature_vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new feature vector with label and optional metadata.
        """
        with self._data_lock:
            self._features.append(feature_vector)
            self._labels.append(label)
            if metadata:
                self._meta[label] = metadata
            self._rebuild_index()

    def query_feature(self, feature_vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Query the nearest neighbor(s) for a given feature vector.
        Returns a list of (label, distance), sorted by increasing distance.
        """
        with self._data_lock:
            if self._nn_index is None or len(self._features) == 0:
                return []

            distances, indices = self._nn_index.kneighbors([feature_vector], n_neighbors=min(top_k, len(self._features)))
            results = [(self._labels[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]
            return results

    def get_metadata(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored metadata for a given label.
        """
        return self._meta.get(label)

    def save_db(self):
        """
        Persist all features, labels, and metadata to disk.
        """
        with self._data_lock:
            db_dict = {
                "features": np.array(self._features, dtype=object),
                "labels": self._labels,
                "meta": self._meta
            }
            with open(LEARNING_DB_FILE, "wb") as f:
                pickle.dump(db_dict, f)
            print(f"[LearningCore] DB saved to {LEARNING_DB_FILE}")

    # ------------------- Internal Helpers ---------------------- #
    def _load_db(self):
        """
        Load persisted data from disk if available.
        """
        if LEARNING_DB_FILE.exists():
            try:
                with open(LEARNING_DB_FILE, "rb") as f:
                    db_dict = pickle.load(f)
                    self._features = list(db_dict.get("features", []))
                    self._labels = db_dict.get("labels", [])
                    self._meta = db_dict.get("meta", {})
                    self._rebuild_index()
                print(f"[LearningCore] Loaded {len(self._labels)} entries from DB.")
            except Exception as e:
                print(f"[LearningCore] Failed to load DB: {e}")

    def _rebuild_index(self):
        """
        Rebuild NearestNeighbors index for fast querying.
        """
        if len(self._features) == 0:
            self._nn_index = None
            return
        try:
            self._nn_index = NearestNeighbors(n_neighbors=min(5, len(self._features)), algorithm="auto", metric="euclidean")
            self._nn_index.fit(self._features)
        except Exception as e:
            print(f"[LearningCore] Failed to rebuild NN index: {e}")

# ------------------------ Factory -------------------------------- #
# Singleton pattern for easy global access
_learning_core_instance: Optional[LearningCore] = None

def get_learning_core() -> LearningCore:
    global _learning_core_instance
    if _learning_core_instance is None:
        _learning_core_instance = LearningCore()
    return _learning_core_instance
