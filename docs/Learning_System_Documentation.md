# Learning System Documentation

## Overview
The Learning System is a comprehensive framework for continuous learning and feature analysis in the OCCUR-CALL AI Camera System. It consists of multiple interconnected modules that enable the system to learn from detected faces, store embeddings, and provide intelligent analysis capabilities.

## Architecture

### Core Components

```
learning/
├── learning_core.py          # Core learning functionality
├── learning_hooks.py          # Hook system for extensibility
├── data_collector.py         # Data collection utilities
├── feature_analyzer.py       # Feature analysis tools
├── incremental_trainer.py    # Incremental learning
├── model_manager.py          # Model management
├── enhancer_connector.py      # AI engine integration
└── learning_hooks.py         # Event-driven analysis
```

## Learning Core (`learning/learning_core.py`)

### Purpose
The LearningCore is the central component that manages feature storage, retrieval, and similarity queries. It provides a unified interface for storing face embeddings and performing nearest neighbor searches.

### Class: LearningCore

#### Constructor
```python
def __init__(self):
    self._data_lock = threading.Lock()           # Thread safety
    self._features: List[np.ndarray] = []       # Feature vectors
    self._labels: List[str] = []                 # Corresponding labels
    self._meta: Dict[str, Dict[str, Any]] = {}  # Metadata per label
    self._nn_index: Optional[NearestNeighbors] = None  # Nearest neighbor index
```

#### Key Methods

##### add_feature()
```python
def add_feature(self, label: str, feature_vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None)
```
**Purpose**: Add a new feature vector with associated label and metadata.

**Parameters**:
- `label`: Unique identifier for the feature
- `feature_vector`: Numpy array representing the feature
- `metadata`: Optional dictionary with additional information

**Process**:
1. Acquire thread lock for safety
2. Append feature vector to storage
3. Append label to labels list
4. Store metadata if provided
5. Rebuild nearest neighbor index

**Example**:
```python
learning_core = get_learning_core()
embedding = np.array([0.1, 0.2, 0.3, ...])  # 128-dimensional vector
metadata = {"timestamp": time.time(), "confidence": 0.95}
learning_core.add_feature("user_123", embedding, metadata)
```

##### query_feature()
```python
def query_feature(self, feature_vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]
```
**Purpose**: Find the most similar features to a given query vector.

**Parameters**:
- `feature_vector`: Query vector to find similarities for
- `top_k`: Number of top results to return

**Returns**:
- List of tuples: `(label, distance)` sorted by increasing distance

**Process**:
1. Acquire thread lock
2. Check if index exists and has data
3. Use NearestNeighbors to find similar features
4. Return results with labels and distances

**Example**:
```python
query_vector = np.array([0.1, 0.2, 0.3, ...])
results = learning_core.query_feature(query_vector, top_k=3)
# Returns: [("user_123", 0.15), ("user_456", 0.23), ("user_789", 0.31)]
```

##### get_metadata()
```python
def get_metadata(self, label: str) -> Optional[Dict[str, Any]]
```
**Purpose**: Retrieve stored metadata for a given label.

**Parameters**:
- `label`: Label to retrieve metadata for

**Returns**:
- Metadata dictionary or None if not found

##### save_db()
```python
def save_db(self)
```
**Purpose**: Persist all learning data to disk using pickle serialization.

**Process**:
1. Acquire thread lock
2. Create data dictionary with features, labels, and metadata
3. Save to pickle file
4. Log success message

**File Location**: `data/learning_db/learning_data.pkl`

### Storage Format
```python
db_dict = {
    "features": np.array(self._features, dtype=object),
    "labels": self._labels,
    "meta": self._meta
}
```

### Thread Safety
- All operations use `threading.Lock()` for thread safety
- Supports concurrent access from multiple threads
- Safe for use in multi-threaded applications

## Learning Hooks (`learning/learning_hooks.py`)

### Purpose
The Learning Hooks system provides an extensible framework for triggering additional analysis modules when face detection events occur. It integrates seamlessly with the AI Engine and LearningCore.

### Hook Registration

#### register_hook()
```python
def register_hook(name: str, func: Callable[..., Any], cooldown: float = DEFAULT_COOLDOWN) -> None
```
**Purpose**: Register a hook function to be triggered by the learning system.

**Parameters**:
- `name`: Unique identifier for the hook
- `func`: Callable function to execute
- `cooldown`: Minimum time between executions (seconds)

**Process**:
1. Validate that function is callable
2. Store hook in HOOKS dictionary
3. Initialize cooldown tracker
4. Set cooldown attribute on function
5. Log registration

**Example**:
```python
def custom_analysis_hook(event_data):
    faces = event_data.get("faces", [])
    print(f"Analyzing {len(faces)} faces")

register_hook("custom_analysis", custom_analysis_hook, cooldown=2.0)
```

#### unregister_hook()
```python
def unregister_hook(name: str) -> None
```
**Purpose**: Remove a previously registered hook.

**Parameters**:
- `name`: Name of hook to remove

### Hook Execution

#### trigger_hooks()
```python
def trigger_hooks(event_data: dict, async_mode: bool = True) -> None
```
**Purpose**: Execute all registered hooks with event data.

**Parameters**:
- `event_data`: Dictionary containing event information
- `async_mode`: Whether to run hooks in separate threads

**Event Data Structure**:
```python
event_data = {
    "frame": np.ndarray,           # Current camera frame
    "faces": List[Dict],           # List of detected faces
    "timestamp": datetime           # Event timestamp
}
```

**Face Data Structure**:
```python
face = {
    "bbox": (x, y, w, h),          # Bounding box
    "user_id": str,                # User identifier
    "confidence": float,           # Recognition confidence
    "saved_snapshot": Path,        # Snapshot path
    "embedding": np.ndarray        # Face embedding
}
```

**Process**:
1. Extract faces and frame from event data
2. Update LearningCore with new embeddings
3. Check cooldown for each hook
4. Execute hooks (async or sync)
5. Update cooldown timestamps

### LearningCore Integration

#### Automatic Embedding Storage
The hook system automatically stores face embeddings in the LearningCore:

```python
for face in faces:
    user_id = face.get("user_id")
    embedding = face.get("embedding")
    metadata = {
        "timestamp": time.time(),
        "bbox": face.get("bbox"),
        "frame_shape": frame.shape if frame is not None else None
    }
    if embedding is not None and user_id is not None:
        _learning_core.add_feature(label=user_id, feature_vector=embedding, metadata=metadata)
```

### Cooldown Management
- **Purpose**: Prevent excessive hook execution
- **Default Cooldown**: 2.0 seconds
- **Per-Hook Cooldown**: Configurable per hook
- **Thread-Safe**: Uses time-based tracking

### Example Hooks

#### Object Detection Hook
```python
def object_detection_hook(event_data: dict) -> None:
    frame = event_data.get("frame")
    if frame is None:
        return
    
    # Perform object detection
    objects = detect_objects(frame)
    print(f"[OBJECT ENGINE] Detected {len(objects)} objects")
    
    # Log results
    for obj in objects:
        log_object_event(obj)

register_hook("object_detection", object_detection_hook, cooldown=1.5)
```

#### Body Analysis Hook
```python
def body_analysis_hook(event_data: dict) -> None:
    faces = event_data.get("faces", [])
    frame = event_data.get("frame")
    
    for face in faces:
        user_id = face.get("user_id")
        bbox = face.get("bbox")
        
        # Perform body analysis
        body_data = analyze_body(frame, bbox)
        
        # Store results
        store_body_analysis(user_id, body_data)

register_hook("body_analysis", body_analysis_hook, cooldown=2.0)
```

#### Custom Analytics Hook
```python
def analytics_hook(event_data: dict) -> None:
    faces = event_data.get("faces", [])
    timestamp = event_data.get("timestamp")
    
    # Perform custom analytics
    analytics_data = {
        "face_count": len(faces),
        "timestamp": timestamp,
        "recognized_faces": [f["user_id"] for f in faces if f["user_id"] != "unknown"],
        "unknown_faces": [f["user_id"] for f in faces if f["user_id"] == "unknown"]
    }
    
    # Send to analytics service
    send_analytics(analytics_data)

register_hook("analytics", analytics_hook, cooldown=5.0)
```

## Data Collection (`learning/data_collector.py`)

### Purpose
The Data Collector module handles systematic collection of training data for the learning system.

### Features
- **Automatic Collection**: Collect data from face detection events
- **Quality Filtering**: Filter out low-quality samples
- **Data Augmentation**: Generate additional training samples
- **Storage Management**: Organize collected data efficiently

### Collection Strategies
1. **Real-time Collection**: Collect from live camera feed
2. **Batch Collection**: Collect from stored images
3. **Incremental Collection**: Add new samples to existing dataset

## Feature Analysis (`learning/feature_analyzer.py`)

### Purpose
The Feature Analyzer provides tools for analyzing and understanding the learned features.

### Analysis Capabilities
- **Feature Visualization**: Visualize high-dimensional features
- **Clustering Analysis**: Group similar features
- **Anomaly Detection**: Identify unusual features
- **Similarity Analysis**: Analyze feature relationships

### Methods
- **PCA Analysis**: Dimensionality reduction for visualization
- **Clustering**: K-means clustering of features
- **Similarity Metrics**: Various distance measures
- **Feature Statistics**: Statistical analysis of features

## Incremental Training (`learning/incremental_trainer.py`)

### Purpose
The Incremental Trainer enables continuous learning by updating models with new data without retraining from scratch.

### Features
- **Online Learning**: Update models with new samples
- **Memory Management**: Manage model memory efficiently
- **Performance Optimization**: Optimize training speed
- **Model Adaptation**: Adapt to changing data distributions

### Training Strategies
1. **Online Learning**: Update models incrementally
2. **Batch Updates**: Update models in batches
3. **Adaptive Learning**: Adjust learning rates dynamically

## Model Management (`learning/model_manager.py`)

### Purpose
The Model Manager handles the lifecycle of machine learning models used in the learning system.

### Features
- **Model Loading**: Load pre-trained models
- **Model Saving**: Save trained models
- **Model Versioning**: Manage model versions
- **Model Evaluation**: Evaluate model performance

### Supported Models
- **Face Recognition Models**: Pre-trained face recognition models
- **Feature Extraction Models**: Models for feature extraction
- **Classification Models**: Models for face classification

## Integration with AI Engine

### Automatic Integration
The learning system automatically integrates with the AI Engine:

```python
# In ai_engine.py
from learning.learning_core import get_learning_core

_learning_core = get_learning_core()

# During face processing
if user_id and embedding is not None:
    _learning_core.add_feature(label=user_id, feature_vector=embedding, metadata={"recognized": True})
```

### Hook Integration
```python
# In camera_main.py
from learning.learning_hooks import trigger_hooks

# After face detection
event_data = {
    "frame": frame,
    "faces": face_events,
    "timestamp": datetime.now()
}
trigger_hooks(event_data, async_mode=True)
```

## Performance Considerations

### Memory Management
- **Efficient Storage**: Use numpy arrays for features
- **Memory Limits**: Implement memory limits for large datasets
- **Garbage Collection**: Proper cleanup of unused data

### Computational Efficiency
- **Index Optimization**: Use efficient nearest neighbor algorithms
- **Batch Processing**: Process multiple features together
- **Caching**: Cache frequently accessed data

### Scalability
- **Distributed Storage**: Support for distributed storage
- **Parallel Processing**: Support for parallel feature processing
- **Load Balancing**: Distribute computational load

## Security Considerations

### Data Privacy
- **Encryption**: Encrypt sensitive feature data
- **Access Control**: Control access to learning data
- **Data Retention**: Implement data retention policies

### Model Security
- **Model Integrity**: Ensure model integrity
- **Adversarial Protection**: Protect against adversarial attacks
- **Secure Storage**: Secure storage of models

## Troubleshooting

### Common Issues

#### Memory Issues
- **Problem**: High memory usage with large datasets
- **Solution**: Implement memory limits and efficient storage

#### Performance Issues
- **Problem**: Slow feature queries
- **Solution**: Optimize nearest neighbor index and use efficient algorithms

#### Data Corruption
- **Problem**: Corrupted learning data
- **Solution**: Implement data validation and backup mechanisms

### Debug Tools
- **Logging**: Comprehensive logging for debugging
- **Metrics**: Performance metrics and monitoring
- **Validation**: Data validation and integrity checks

## Best Practices

### Data Management
1. **Regular Backups**: Backup learning data regularly
2. **Data Validation**: Validate data before storage
3. **Quality Control**: Implement quality filters
4. **Version Control**: Version control for learning data

### Performance Optimization
1. **Efficient Algorithms**: Use efficient algorithms for feature processing
2. **Memory Management**: Implement proper memory management
3. **Caching**: Use caching for frequently accessed data
4. **Parallel Processing**: Utilize parallel processing where possible

### Security
1. **Access Control**: Implement proper access control
2. **Encryption**: Encrypt sensitive data
3. **Audit Logging**: Implement audit logging
4. **Regular Updates**: Keep learning system updated

---

This documentation provides comprehensive coverage of the Learning System's architecture, functionality, and usage patterns. The system is designed to be extensible, efficient, and secure, providing a solid foundation for continuous learning in the AI Camera system.
