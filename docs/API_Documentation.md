# API Documentation

## Overview
This document provides comprehensive API documentation for the OCCUR-CALL AI Camera System. It covers all public interfaces, methods, and usage patterns for developers integrating with the system.

## Table of Contents
1. [Core API](#core-api)
2. [AI Engine API](#ai-engine-api)
3. [Learning System API](#learning-system-api)
4. [Database API](#database-api)
5. [Storage API](#storage-api)
6. [Hook System API](#hook-system-api)
7. [Configuration API](#configuration-api)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

## Core API

### AICameraEngine

The main engine class for face detection and recognition.

#### Constructor
```python
AICameraEngine(
    recognizer: str = "auto",
    every_n_frames: int = 1,
    draw: bool = True,
    unknown_cooldown: float = 15.0,
    snapshot_cooldown: float = 5.0,
    enable_db_logging: bool = True
) -> AICameraEngine
```

**Parameters**:
- `recognizer` (str): Recognition backend ("auto", "face_recognition", "lbph", "none")
- `every_n_frames` (int): Process every N frames for performance
- `draw` (bool): Draw bounding boxes on frames
- `unknown_cooldown` (float): Cooldown for unknown face saves (seconds)
- `snapshot_cooldown` (float): Cooldown for snapshot saves (seconds)
- `enable_db_logging` (bool): Enable database logging

**Returns**: AICameraEngine instance

**Example**:
```python
from engines.ai_engine import AICameraEngine

# Create engine with default settings
ai = AICameraEngine()

# Create engine with custom settings
ai = AICameraEngine(
    recognizer="face_recognition",
    every_n_frames=2,
    unknown_cooldown=30.0
)
```

#### detect_and_recognize()
```python
def detect_and_recognize(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]
```

**Purpose**: Main processing method for face detection and recognition.

**Parameters**:
- `frame_bgr` (np.ndarray): Input camera frame in BGR format

**Returns**:
- `annotated` (np.ndarray): Frame with bounding boxes and labels
- `face_events` (List[Dict]): List of detected faces with metadata

**Face Event Structure**:
```python
{
    "bbox": (x, y, w, h),           # Bounding box coordinates
    "user_id": str,                # User ID or "unknown"
    "confidence": float,            # Recognition confidence (0.0-1.0)
    "saved_snapshot": Path,         # Path to saved snapshot
    "embedding": np.ndarray         # Face embedding vector
}
```

**Example**:
```python
import cv2

cap = cv2.VideoCapture(0)
ai = AICameraEngine()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated_frame, face_events = ai.detect_and_recognize(frame)
    
    # Process face events
    for face in face_events:
        print(f"Detected: {face['user_id']} with confidence {face['confidence']}")
    
    cv2.imshow("AI Camera", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

#### register_user()
```python
def register_user(self, user_id: str, face_bgr_image: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None) -> Path
```

**Purpose**: Register a new user by saving their face image.

**Parameters**:
- `user_id` (str): Unique identifier for the user
- `face_bgr_image` (np.ndarray): Face image in BGR format
- `bbox` (Optional[Tuple]): Optional bounding box (x, y, w, h)

**Returns**: Path to saved face image

**Example**:
```python
# Register user from camera frame
ret, frame = cap.read()
faces = detect_faces(frame)
if faces:
    x, y, w, h = faces[0]
    saved_path = ai.register_user("john_doe", frame, (x, y, w, h))
    print(f"User registered: {saved_path}")

# Register user from image file
import cv2
image = cv2.imread("user_photo.jpg")
saved_path = ai.register_user("jane_doe", image)
```

#### reload_models()
```python
def reload_models(self) -> None
```

**Purpose**: Reload all recognition models from the face database.

**Example**:
```python
# Reload models after adding new users
ai.reload_models()
print("Models reloaded successfully")
```

### Factory Function

#### create_engine()
```python
def create_engine(
    recognizer: str = "auto",
    every_n_frames: int = 1,
    draw: bool = True,
    unknown_cooldown: float = 15.0,
    snapshot_cooldown: float = 5.0,
    enable_db_logging: bool = True
) -> AICameraEngine
```

**Purpose**: Factory function to create AICameraEngine instances.

**Example**:
```python
from engines.ai_engine import create_engine

# Create engine using factory function
ai = create_engine(recognizer="face_recognition", every_n_frames=2)
```

## Learning System API

### LearningCore

Core learning functionality for feature storage and retrieval.

#### get_learning_core()
```python
def get_learning_core() -> LearningCore
```

**Purpose**: Get singleton instance of LearningCore.

**Returns**: LearningCore instance

**Example**:
```python
from learning.learning_core import get_learning_core

learning_core = get_learning_core()
```

#### add_feature()
```python
def add_feature(self, label: str, feature_vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None
```

**Purpose**: Add a new feature vector with label and metadata.

**Parameters**:
- `label` (str): Unique identifier for the feature
- `feature_vector` (np.ndarray): Feature vector (e.g., face embedding)
- `metadata` (Optional[Dict]): Additional metadata

**Example**:
```python
import numpy as np

# Add face embedding
embedding = np.array([0.1, 0.2, 0.3, ...])  # 128-dimensional vector
metadata = {
    "timestamp": time.time(),
    "confidence": 0.95,
    "source": "camera"
}
learning_core.add_feature("user_123", embedding, metadata)
```

#### query_feature()
```python
def query_feature(self, feature_vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]
```

**Purpose**: Find the most similar features to a query vector.

**Parameters**:
- `feature_vector` (np.ndarray): Query vector
- `top_k` (int): Number of top results to return

**Returns**: List of tuples (label, distance) sorted by distance

**Example**:
```python
# Query similar features
query_vector = np.array([0.1, 0.2, 0.3, ...])
results = learning_core.query_feature(query_vector, top_k=3)

for label, distance in results:
    print(f"Similar to {label} with distance {distance}")
```

#### get_metadata()
```python
def get_metadata(self, label: str) -> Optional[Dict[str, Any]]
```

**Purpose**: Retrieve metadata for a given label.

**Parameters**:
- `label` (str): Label to retrieve metadata for

**Returns**: Metadata dictionary or None

**Example**:
```python
metadata = learning_core.get_metadata("user_123")
if metadata:
    print(f"Last seen: {metadata['timestamp']}")
```

#### save_db()
```python
def save_db(self) -> None
```

**Purpose**: Persist learning data to disk.

**Example**:
```python
# Save learning data
learning_core.save_db()
print("Learning data saved successfully")
```

### Learning Hooks

Hook system for extensible analysis.

#### register_hook()
```python
def register_hook(name: str, func: Callable[..., Any], cooldown: float = DEFAULT_COOLDOWN) -> None
```

**Purpose**: Register a hook function for event-driven analysis.

**Parameters**:
- `name` (str): Unique hook identifier
- `func` (Callable): Hook function to execute
- `cooldown` (float): Minimum time between executions (seconds)

**Example**:
```python
from learning.learning_hooks import register_hook

def custom_analysis_hook(event_data):
    faces = event_data.get("faces", [])
    frame = event_data.get("frame")
    
    # Perform custom analysis
    for face in faces:
        user_id = face.get("user_id")
        bbox = face.get("bbox")
        print(f"Analyzing face for user {user_id}")

# Register the hook
register_hook("custom_analysis", custom_analysis_hook, cooldown=2.0)
```

#### trigger_hooks()
```python
def trigger_hooks(event_data: dict, async_mode: bool = True) -> None
```

**Purpose**: Execute all registered hooks with event data.

**Parameters**:
- `event_data` (dict): Event data dictionary
- `async_mode` (bool): Run hooks in separate threads

**Event Data Structure**:
```python
event_data = {
    "frame": np.ndarray,           # Current camera frame
    "faces": List[Dict],           # List of detected faces
    "timestamp": datetime          # Event timestamp
}
```

**Example**:
```python
from learning.learning_hooks import trigger_hooks

# Trigger hooks after face detection
event_data = {
    "frame": current_frame,
    "faces": detected_faces,
    "timestamp": datetime.now()
}
trigger_hooks(event_data, async_mode=True)
```

#### query_learning_system()
```python
def query_learning_system(embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]
```

**Purpose**: Query the learning system for similar features.

**Parameters**:
- `embedding` (np.ndarray): Query embedding
- `top_k` (int): Number of top results

**Returns**: List of (label, distance) tuples

**Example**:
```python
from learning.learning_hooks import query_learning_system

# Query learning system
results = query_learning_system(face_embedding, top_k=5)
for label, distance in results:
    print(f"Similar to {label} with distance {distance}")
```

## Database API

### Database Utilities

Database operations and logging.

#### log_face_event()
```python
def log_face_event(event_type: str, user_id: str = None, image_path: str = None, confidence: float = None) -> None
```

**Purpose**: Log face detection and recognition events.

**Parameters**:
- `event_type` (str): Event type ("recognized", "unknown", "auto_register")
- `user_id` (str): User identifier (optional)
- `image_path` (str): Path to saved image (optional)
- `confidence` (float): Recognition confidence (optional)

**Example**:
```python
from Utils.db_utils import log_face_event

# Log recognized face
log_face_event(
    event_type="recognized",
    user_id="john_doe",
    confidence=0.95
)

# Log unknown face
log_face_event(
    event_type="unknown",
    image_path="/path/to/unknown_face.jpg"
)
```

#### log_system_event()
```python
def log_system_event(event_type: str, details: str = None) -> None
```

**Purpose**: Log system events and status changes.

**Parameters**:
- `event_type` (str): System event type
- `details` (str): Additional event details (optional)

**Example**:
```python
from Utils.db_utils import log_system_event

# Log system startup
log_system_event(
    event_type="system_start",
    details="AI Camera system started successfully"
)

# Log error
log_system_event(
    event_type="error",
    details="Camera initialization failed"
)
```

## Storage API

### Snapshot Management

#### save_unknown_snapshot()
```python
def save_unknown_snapshot(face_img: np.ndarray) -> str
```

**Purpose**: Save unknown face snapshot and return path.

**Parameters**:
- `face_img` (np.ndarray): Face image to save

**Returns**: Path to saved snapshot

**Example**:
```python
from camera_main import save_unknown_snapshot

# Save unknown face
face_crop = frame[y:y+h, x:x+w]
snapshot_path = save_unknown_snapshot(face_crop)
print(f"Unknown face saved: {snapshot_path}")
```

## Configuration API

### Configuration Management

#### Configuration Variables
```python
# Base directories
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
FACE_DB_DIR = DATA_DIR / "face_db"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
UNKNOWN_DIR = DATA_DIR / "unknown_faces"
MODEL_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "Update_and_Backup"

# Database paths
OCCUR_DB = Path(r"C:\OCCUR-CALL\Backend\instance\occur-call.db")
DB_PATH = DB_DIR / "face_events.db"
```

**Example**:
```python
from config import FACE_DB_DIR, SNAPSHOT_DIR

# Use configuration variables
user_dir = FACE_DB_DIR / "user_123"
snapshot_path = SNAPSHOT_DIR / "snapshot.jpg"
```

## Error Handling

### Common Exceptions

#### AICameraEngine Exceptions
```python
class AIEngineError(Exception):
    """Base exception for AI Engine errors."""
    pass

class RecognitionError(AIEngineError):
    """Recognition-related errors."""
    pass

class DatabaseError(AIEngineError):
    """Database-related errors."""
    pass
```

### Error Handling Patterns

#### Try-Catch Blocks
```python
try:
    ai = AICameraEngine(recognizer="face_recognition")
    annotated_frame, face_events = ai.detect_and_recognize(frame)
except RecognitionError as e:
    print(f"Recognition error: {e}")
    # Fallback to detection-only mode
    ai = AICameraEngine(recognizer="none")
except DatabaseError as e:
    print(f"Database error: {e}")
    # Continue without database logging
    ai.enable_db_logging = False
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

#### Graceful Degradation
```python
def safe_face_recognition(frame):
    """Safe face recognition with fallback options."""
    try:
        # Try face recognition
        ai = AICameraEngine(recognizer="face_recognition")
        return ai.detect_and_recognize(frame)
    except RecognitionError:
        try:
            # Fallback to LBPH
            ai = AICameraEngine(recognizer="lbph")
            return ai.detect_and_recognize(frame)
        except RecognitionError:
            # Fallback to detection-only
            ai = AICameraEngine(recognizer="none")
            return ai.detect_and_recognize(frame)
```

## Examples

### Basic Face Detection and Recognition

```python
import cv2
from engines.ai_engine import AICameraEngine

def main():
    # Initialize AI engine
    ai = AICameraEngine(recognizer="auto")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and recognize faces
        annotated_frame, face_events = ai.detect_and_recognize(frame)
        
        # Process face events
        for face in face_events:
            user_id = face["user_id"]
            confidence = face.get("confidence")
            bbox = face["bbox"]
            
            if user_id and user_id != "unknown":
                print(f"Recognized: {user_id} (confidence: {confidence})")
            else:
                print("Unknown face detected")
        
        # Display frame
        cv2.imshow("AI Camera", annotated_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Custom Hook Implementation

```python
from learning.learning_hooks import register_hook, trigger_hooks
from learning.learning_core import get_learning_core
import time

# Initialize learning core
learning_core = get_learning_core()

def body_analysis_hook(event_data):
    """Custom hook for body analysis."""
    faces = event_data.get("faces", [])
    frame = event_data.get("frame")
    
    for face in faces:
        user_id = face.get("user_id")
        bbox = face.get("bbox")
        embedding = face.get("embedding")
        
        if embedding is not None:
            # Perform body analysis
            body_data = analyze_body_shape(frame, bbox)
            
            # Store analysis results
            metadata = {
                "body_analysis": body_data,
                "timestamp": time.time()
            }
            learning_core.add_feature(f"{user_id}_body", embedding, metadata)

def object_detection_hook(event_data):
    """Custom hook for object detection."""
    frame = event_data.get("frame")
    
    if frame is not None:
        # Perform object detection
        objects = detect_objects(frame)
        
        # Log detected objects
        for obj in objects:
            print(f"Detected object: {obj['type']} at {obj['bbox']}")

# Register hooks
register_hook("body_analysis", body_analysis_hook, cooldown=2.0)
register_hook("object_detection", object_detection_hook, cooldown=1.0)

# Use in main loop
def main():
    ai = AICameraEngine()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, face_events = ai.detect_and_recognize(frame)
        
        # Trigger custom hooks
        event_data = {
            "frame": frame,
            "faces": face_events,
            "timestamp": time.time()
        }
        trigger_hooks(event_data, async_mode=True)
        
        cv2.imshow("AI Camera", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### Database Integration

```python
from Utils.db_utils import log_face_event, log_system_event
from engines.ai_engine import AICameraEngine
import cv2

def main():
    # Log system startup
    log_system_event("system_start", "AI Camera system started")
    
    # Initialize AI engine
    ai = AICameraEngine(enable_db_logging=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, face_events = ai.detect_and_recognize(frame)
            
            # Log face events
            for face in face_events:
                user_id = face["user_id"]
                confidence = face.get("confidence")
                image_path = face.get("saved_snapshot")
                
                if user_id and user_id != "unknown":
                    log_face_event(
                        event_type="recognized",
                        user_id=user_id,
                        confidence=confidence
                    )
                else:
                    log_face_event(
                        event_type="unknown",
                        image_path=str(image_path) if image_path else None
                    )
            
            cv2.imshow("AI Camera", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        log_system_event("error", f"System error: {str(e)}")
        raise
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_system_event("system_stop", "AI Camera system stopped")
```

### Learning System Integration

```python
from learning.learning_core import get_learning_core
from learning.learning_hooks import register_hook, trigger_hooks
from engines.ai_engine import AICameraEngine
import numpy as np

def main():
    # Initialize learning core
    learning_core = get_learning_core()
    
    # Initialize AI engine
    ai = AICameraEngine()
    
    # Custom learning hook
    def learning_hook(event_data):
        faces = event_data.get("faces", [])
        
        for face in faces:
            user_id = face.get("user_id")
            embedding = face.get("embedding")
            
            if embedding is not None and user_id:
                # Add to learning core
                metadata = {
                    "timestamp": time.time(),
                    "confidence": face.get("confidence"),
                    "bbox": face.get("bbox")
                }
                learning_core.add_feature(user_id, embedding, metadata)
                
                # Query similar features
                similar = learning_core.query_feature(embedding, top_k=3)
                print(f"Similar features for {user_id}: {similar}")
    
    # Register learning hook
    register_hook("learning", learning_hook, cooldown=1.0)
    
    # Main loop
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, face_events = ai.detect_and_recognize(frame)
        
        # Trigger learning hooks
        event_data = {
            "frame": frame,
            "faces": face_events,
            "timestamp": time.time()
        }
        trigger_hooks(event_data, async_mode=True)
        
        cv2.imshow("AI Camera", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save learning data
    learning_core.save_db()
    
    cap.release()
    cv2.destroyAllWindows()
```

## Best Practices

### Performance Optimization
1. **Frame Processing**: Use `every_n_frames` parameter for performance
2. **Async Hooks**: Use async mode for hook execution
3. **Memory Management**: Properly manage memory for large datasets
4. **Caching**: Cache frequently accessed data

### Error Handling
1. **Graceful Degradation**: Implement fallback mechanisms
2. **Exception Handling**: Use try-catch blocks appropriately
3. **Logging**: Log errors for debugging
4. **Recovery**: Implement recovery mechanisms

### Security
1. **Input Validation**: Validate all inputs
2. **Access Control**: Implement proper access controls
3. **Data Encryption**: Encrypt sensitive data
4. **Audit Logging**: Log all system activities

### Code Quality
1. **Documentation**: Document all functions and classes
2. **Testing**: Write comprehensive tests
3. **Code Review**: Regular code reviews
4. **Standards**: Follow coding standards

---

This API documentation provides comprehensive coverage of all public interfaces in the OCCUR-CALL AI Camera System. For additional information, refer to the individual module documentation and inline code comments.
