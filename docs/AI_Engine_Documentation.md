# AI Engine Documentation

## Overview
The AI Engine (`engines/ai_engine.py`) is the core component of the OCCUR-CALL AI Camera System, responsible for real-time face detection and recognition. It provides a unified interface for multiple recognition backends and integrates with the learning system for continuous improvement.

## Architecture

### Class: AICameraEngine

The main engine class that handles all face detection and recognition operations.

#### Constructor Parameters
```python
def __init__(
    self,
    recognizer: str = "auto",           # Recognition backend: "auto", "face_recognition", "lbph", "none"
    every_n_frames: int = 1,            # Process every N frames for performance
    draw: bool = True,                  # Draw bounding boxes on frames
    unknown_cooldown: float = 15.0,     # Cooldown for unknown face saves (seconds)
    snapshot_cooldown: float = 5.0,     # Cooldown for snapshot saves (seconds)
    enable_db_logging: bool = True      # Enable database logging
) -> None
```

#### Core Methods

##### detect_and_recognize()
```python
def detect_and_recognize(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]
```
**Purpose**: Main processing method that detects faces and performs recognition.

**Parameters**:
- `frame_bgr`: Input camera frame in BGR format

**Returns**:
- `annotated`: Frame with bounding boxes and labels drawn
- `face_events`: List of detected faces with metadata

**Process Flow**:
1. Convert frame to grayscale and apply histogram equalization
2. Detect faces using Haar cascade
3. For each detected face:
   - Extract face embedding
   - Perform recognition (if enabled)
   - Handle unknown faces
   - Draw bounding boxes
   - Create face event data

**Face Event Structure**:
```python
{
    "bbox": (x, y, w, h),           # Bounding box coordinates
    "user_id": str,                 # Recognized user ID or "unknown"
    "confidence": float,             # Recognition confidence (0.0-1.0)
    "saved_snapshot": Path,         # Path to saved snapshot (if unknown)
    "embedding": np.ndarray         # Face embedding vector
}
```

##### register_user()
```python
def register_user(self, user_id: str, face_bgr_image: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None) -> Path
```
**Purpose**: Register a new user by saving their face image and retraining models.

**Parameters**:
- `user_id`: Unique identifier for the user
- `face_bgr_image`: Face image in BGR format
- `bbox`: Optional bounding box (x, y, w, h) to crop face

**Returns**:
- `Path`: Path to saved face image

**Process**:
1. Create user directory in face database
2. Crop face from image if bbox provided
3. Save face image with timestamp
4. Reload recognition models

##### reload_models()
```python
def reload_models(self) -> None
```
**Purpose**: Reload all recognition models from the face database.

**Process**:
1. Reinitialize recognition backend
2. Load all face images from database
3. Train recognition models
4. Update known encodings/labels

## Recognition Backends

### 1. Face Recognition Backend
**Library**: `face_recognition` (preferred)
**Method**: `_try_init_face_recognition()`

**Features**:
- Uses HOG (Histogram of Oriented Gradients) for face detection
- Generates 128-dimensional face encodings
- High accuracy and robustness
- Supports multiple face orientations

**Recognition Process**:
1. Load face image using `face_recognition.load_image_file()`
2. Detect face locations using HOG model
3. Extract face encodings
4. Compare with known encodings using Euclidean distance
5. Return user ID if distance ≤ 0.6

**Advantages**:
- High accuracy
- Robust to lighting changes
- Good performance
- Well-maintained library

**Disadvantages**:
- Requires additional dependencies
- Larger memory footprint
- Slower initialization

### 2. LBPH Backend
**Library**: OpenCV `cv2.face.LBPHFaceRecognizer_create()`
**Method**: `_try_init_lbph()`

**Features**:
- Local Binary Pattern Histogram algorithm
- Built into OpenCV
- Lightweight and fast
- Good for controlled environments

**Recognition Process**:
1. Convert images to grayscale
2. Resize faces to 200x200 pixels
3. Train LBPH recognizer
4. Predict using confidence threshold ≤ 80

**Advantages**:
- No external dependencies
- Fast training and recognition
- Low memory usage
- Built into OpenCV

**Disadvantages**:
- Lower accuracy than face_recognition
- Sensitive to lighting changes
- Limited to frontal faces

### 3. Detection-Only Mode
**Method**: `_set_none()`

**Features**:
- Face detection without recognition
- Minimal resource usage
- Fallback when recognition fails

**Use Cases**:
- Testing face detection
- Resource-constrained environments
- When recognition libraries unavailable

## Face Detection

### Haar Cascade Detection
**File**: `haarcascade_frontalface_default.xml`
**Method**: `detectMultiScale()`

**Parameters**:
- `scaleFactor=1.2`: Scale factor for image pyramid
- `minNeighbors=5`: Minimum neighbors for face detection
- `minSize=(60, 60)`: Minimum face size

**Process**:
1. Convert frame to grayscale
2. Apply histogram equalization for lighting normalization
3. Run Haar cascade detection
4. Return bounding boxes for detected faces

## Unknown Face Handling

### Process Flow
1. **Detection**: Face detected but not recognized
2. **Cooldown Check**: Prevent duplicate saves using face hash
3. **Image Enhancement**: Apply lighting normalization
4. **Snapshot Save**: Save unknown face image with timestamp
5. **Auto-Registration**: Register as temporary user
6. **Database Logging**: Log unknown face event
7. **Learning Integration**: Add to learning core

### Cooldown System
**Purpose**: Prevent duplicate saves of the same unknown face
**Method**: Face hashing using SHA1 of resized grayscale image
**Default Cooldown**: 15 seconds

```python
def _handle_unknown(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, Optional[float], Optional[Path], Optional[np.ndarray]]
```

## Embedding Extraction

### Purpose
Extract fixed-size feature vectors for face learning and similarity comparison.

### Methods by Backend

#### Face Recognition Backend
- **Vector Size**: 128 dimensions
- **Method**: `face_recognition.face_encodings()`
- **Format**: Float32 array

#### LBPH Backend
- **Vector Size**: 4096 dimensions (64x64 flattened)
- **Method**: Resize to 64x64, flatten, normalize
- **Format**: Normalized float32 array

#### Fallback Method
- **Vector Size**: 4096 dimensions
- **Method**: Same as LBPH backend
- **Format**: Normalized float32 array

## Database Integration

### Face Events Table
```sql
CREATE TABLE face_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    user_id TEXT,
    image_path TEXT,
    confidence REAL
)
```

### Event Types
- `recognized`: Known face detected
- `unknown`: Unknown face detected
- `auto_register`: Unknown face auto-registered

### Logging Methods
- `_log_face_event_db()`: Internal logging method
- `log_face_event()`: Public logging interface (from db_utils.py)

## Performance Optimization

### Frame Processing
- **every_n_frames**: Process every N frames instead of every frame
- **Threaded Recognition**: Optional threaded processing
- **Cooldown Management**: Prevent duplicate processing

### Memory Management
- **Lazy Loading**: Load models only when needed
- **Efficient Storage**: Use numpy arrays for embeddings
- **Cleanup**: Proper resource cleanup on exit

### Recognition Optimization
- **Histogram Equalization**: Improve lighting conditions
- **Multi-scale Detection**: Handle different face sizes
- **Confidence Thresholding**: Filter low-confidence matches

## Error Handling

### Common Errors
1. **Camera Access**: Handle camera initialization failures
2. **Model Loading**: Graceful fallback when models fail
3. **Database Errors**: Continue operation despite DB failures
4. **Recognition Failures**: Fallback to detection-only mode

### Error Recovery
- **Automatic Fallback**: Switch to alternative backends
- **Graceful Degradation**: Continue with reduced functionality
- **Error Logging**: Log errors for debugging
- **User Notification**: Inform users of system status

## Integration Points

### Learning System
- **Embedding Storage**: Automatic embedding extraction
- **Feature Learning**: Integration with LearningCore
- **Continuous Improvement**: Learn from new faces

### Hook System
- **Event Triggers**: Trigger analysis hooks on face detection
- **Extensible Architecture**: Support for custom analysis modules
- **Async Processing**: Non-blocking hook execution

### Storage System
- **Snapshot Management**: Automatic snapshot saving
- **Database Logging**: Event logging to SQLite
- **Backup Integration**: Integration with backup system

## Configuration Options

### Recognition Backend Selection
```python
# Auto-select best available backend
ai = AICameraEngine(recognizer="auto")

# Force specific backend
ai = AICameraEngine(recognizer="face_recognition")
ai = AICameraEngine(recognizer="lbph")
ai = AICameraEngine(recognizer="none")
```

### Performance Tuning
```python
# Process every 3rd frame for better performance
ai = AICameraEngine(every_n_frames=3)

# Disable drawing for headless operation
ai = AICameraEngine(draw=False)

# Adjust cooldowns
ai = AICameraEngine(unknown_cooldown=30.0, snapshot_cooldown=10.0)
```

### Database Configuration
```python
# Disable database logging
ai = AICameraEngine(enable_db_logging=False)
```

## Best Practices

### Face Database Management
1. **Image Quality**: Use high-quality, well-lit face images
2. **Multiple Angles**: Include different angles and expressions
3. **Regular Updates**: Retrain models when adding new users
4. **Backup**: Regular backup of face database

### Performance Optimization
1. **Frame Rate**: Adjust `every_n_frames` based on system performance
2. **Resolution**: Use appropriate camera resolution
3. **Lighting**: Ensure good lighting conditions
4. **Distance**: Maintain appropriate face-to-camera distance

### Security Considerations
1. **Access Control**: Secure face database access
2. **Data Privacy**: Handle face data according to privacy laws
3. **Encryption**: Consider encrypting sensitive data
4. **Audit Logging**: Monitor system usage

## Troubleshooting

### Recognition Issues
- **Low Accuracy**: Check image quality and lighting
- **No Recognition**: Verify face database and model loading
- **False Positives**: Adjust confidence thresholds
- **Performance**: Reduce `every_n_frames` or use detection-only mode

### System Issues
- **Camera Access**: Check camera permissions and drivers
- **Database Errors**: Verify file permissions and disk space
- **Memory Issues**: Reduce face database size or use LBPH backend
- **Dependency Issues**: Ensure all required libraries are installed

---

This documentation provides comprehensive coverage of the AI Engine's functionality, architecture, and usage patterns. For additional information, refer to the inline code comments and the main README.md file.
