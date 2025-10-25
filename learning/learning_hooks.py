"""
learning_hooks.py — OCCUR-CALL Learning Hook System (Final)

Purpose
-------
- Central system to trigger additional engines or analysis modules
- Integrates smoothly with AI Engine without modifying it
- Supports async hooks, cooldowns, and prioritized execution
- Extensible: can add object_engine, body_engine, auth_engine, or any custom engine
- Integrated with LearningCore for storing/querying embeddings
"""

from __future__ import annotations
from typing import Callable, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import threading
import time
from learning.learning_core import get_learning_core

# ----------------------- LearningCore Integration ----------------------- #
_learning_core = get_learning_core()

# ----------------------------- Hook Registry ----------------------------- #
# Dictionary: hook_name -> hook function
HOOKS: Dict[str, Callable[..., Any]] = {}

# Cooldown tracker: hook_name -> last_execution_timestamp
HOOK_COOLDOWNS: Dict[str, float] = {}

# Default cooldown per hook in seconds
DEFAULT_COOLDOWN: float = 2.0

# -------------------------- Public API ---------------------------------- #
def register_hook(name: str, func: Callable[..., Any], cooldown: float = DEFAULT_COOLDOWN) -> None:
    """
    Register a hook function to be triggered by the learning system.

    Parameters:
    - name: unique name for the hook
    - func: callable to execute when triggered
    - cooldown: minimum time in seconds between consecutive executions
    """
    if not callable(func):
        raise ValueError(f"Hook '{name}' must be callable.")
    HOOKS[name] = func
    HOOK_COOLDOWNS[name] = -float('inf')
    setattr(func, "_cooldown", cooldown)
    print(f"[HOOK] Registered hook '{name}' with cooldown {cooldown}s.")

def unregister_hook(name: str) -> None:
    """Remove a previously registered hook."""
    if name in HOOKS:
        del HOOKS[name]
        del HOOK_COOLDOWNS[name]
        print(f"[HOOK] Unregistered hook '{name}'.")

# ----------------------- Internal Safe Execution ------------------------ #
def _run_hook_safe(name: str, func: Callable[..., Any], event_data: dict) -> None:
    """Call a hook safely and catch exceptions."""
    try:
        func(event_data)
    except Exception as e:
        print(f"[HOOK][ERROR] Hook '{name}' failed: {e}")

# ----------------------- Trigger Hooks & Learning ------------------------ #
def trigger_hooks(event_data: dict, async_mode: bool = True) -> None:
    """
    Trigger all registered hooks with event_data.
    Integrates LearningCore to store face embeddings automatically.
    
    Parameters:
    - event_data: dictionary of context information (frame, faces, timestamp, etc.)
    - async_mode: if True, each hook runs in a separate thread
    """
    now_ts = time.time()
    faces = event_data.get("faces", [])
    frame = event_data.get("frame")

    # -----------------------
    # Update LearningCore
    # -----------------------
    for face in faces:
        user_id = face.get("user_id")
        embedding = face.get("embedding")  # Must be provided by AI engine
        metadata = {
            "timestamp": time.time(),
            "bbox": face.get("bbox"),
            "frame_shape": frame.shape if frame is not None else None
        }
        if embedding is not None and user_id is not None:
            _learning_core.add_feature(label=user_id, feature_vector=embedding, metadata=metadata)

    # -----------------------
    # Trigger all registered hooks with cooldown logic
    # -----------------------
    for name, func in HOOKS.items():
        last_run = HOOK_COOLDOWNS.get(name, -float('inf'))
        cooldown = getattr(func, "_cooldown", DEFAULT_COOLDOWN)
        if now_ts - last_run >= cooldown:
            if async_mode:
                threading.Thread(target=_run_hook_safe, args=(name, func, event_data), daemon=True).start()
            else:
                _run_hook_safe(name, func, event_data)
            HOOK_COOLDOWNS[name] = now_ts

# ----------------------- Learning Query Helper --------------------------- #
def query_learning_system(embedding, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Query LearningCore for nearest neighbors of a given embedding.
    
    Returns:
    - List of tuples: (label, distance), sorted by nearest distance
    """
    return _learning_core.query_feature(embedding, top_k=top_k)

# ----------------------- Example Hook Templates ------------------------- #
def example_object_engine_hook(event_data: dict) -> None:
    """Example hook for object detection engine."""
    frame = event_data.get("frame")
    if frame is None:
        return
    # TODO: integrate actual object detection engine
    # detected_objects = object_engine.detect(frame)
    # print(f"[OBJECT ENGINE] Detected {len(detected_objects)} objects.")

def example_body_engine_hook(event_data: dict) -> None:
    """Example hook for body analysis engine."""
    faces = event_data.get("faces", [])
    for face in faces:
        user_id = face.get("user_id")
        bbox = face.get("bbox")
        # TODO: integrate body engine
        # result = body_engine.analyze(face, bbox)
        # print(f"[BODY ENGINE] Analysis for {user_id}: {result}")

# ---------------------- Auto-Registration of Example Hooks --------------- #
register_hook("object_engine", example_object_engine_hook, cooldown=1.5)
register_hook("body_engine", example_body_engine_hook, cooldown=2.0)
