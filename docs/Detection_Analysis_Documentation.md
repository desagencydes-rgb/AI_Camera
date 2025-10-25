# Detection and Analysis Modules Documentation

## Overview
The Detection and Analysis modules provide specialized computer vision capabilities for the OCCUR-CALL AI Camera System. These modules handle various aspects of human detection, analysis, and monitoring beyond basic face recognition.

## Detection Modules (`detection/`)

### Face Detection (`detection/face_detection.py`)

#### Purpose
Core face detection utilities using OpenCV Haar cascades and advanced detection techniques.

#### Features
- **Haar Cascade Detection**: Primary face detection method
- **Multi-scale Detection**: Handle faces of different sizes
- **Lighting Adaptation**: Adapt to various lighting conditions
- **Performance Optimization**: Optimized for real-time processing

#### Key Functions
```python
def detect_faces(frame, cascade_path=None):
    """
    Detect faces in a frame using Haar cascades.
    
    Parameters:
    - frame: Input image frame
    - cascade_path: Optional path to custom cascade file
    
    Returns:
    - List of bounding boxes (x, y, w, h)
    """
```

#### Detection Parameters
- **Scale Factor**: 1.2 (controls image pyramid scaling)
- **Min Neighbors**: 5 (minimum neighbors for detection)
- **Min Size**: (60, 60) (minimum face size)
- **Max Size**: None (no maximum size limit)

#### Performance Optimization
- **Histogram Equalization**: Improve lighting conditions
- **Multi-threading**: Parallel processing for multiple scales
- **Caching**: Cache cascade classifiers for reuse

### Eye Detection (`detection/eye_detection.py`)

#### Purpose
Detect and track human eyes for advanced analysis and attention monitoring.

#### Features
- **Eye Region Detection**: Locate eye regions within faces
- **Eye State Analysis**: Determine if eyes are open or closed
- **Gaze Tracking**: Track eye movement and direction
- **Blink Detection**: Detect and count blinks

#### Detection Methods
1. **Haar Cascade**: Using `haarcascade_eye.xml`
2. **HOG Descriptor**: Histogram of Oriented Gradients
3. **Deep Learning**: Pre-trained eye detection models

#### Key Functions
```python
def detect_eyes(face_region):
    """
    Detect eyes within a face region.
    
    Parameters:
    - face_region: Cropped face image
    
    Returns:
    - List of eye bounding boxes
    """

def analyze_eye_state(eye_region):
    """
    Analyze the state of detected eyes.
    
    Parameters:
    - eye_region: Cropped eye image
    
    Returns:
    - Dictionary with eye state information
    """
```

#### Applications
- **Attention Monitoring**: Track user attention
- **Fatigue Detection**: Detect drowsiness or fatigue
- **Accessibility**: Assist users with visual impairments
- **Security**: Detect suspicious behavior

### Pose Detection (`detection/pose_detection.py`)

#### Purpose
Detect and analyze human pose for body language analysis and activity recognition.

#### Features
- **Keypoint Detection**: Detect body keypoints
- **Pose Estimation**: Estimate human pose
- **Activity Recognition**: Recognize human activities
- **Gesture Analysis**: Analyze hand gestures

#### Detection Methods
1. **OpenPose**: Real-time multi-person pose estimation
2. **MediaPipe**: Google's pose detection framework
3. **YOLO-Pose**: YOLO-based pose detection
4. **Custom Models**: Custom-trained pose models

#### Key Functions
```python
def detect_pose(frame):
    """
    Detect human pose in a frame.
    
    Parameters:
    - frame: Input image frame
    
    Returns:
    - List of pose keypoints
    """

def analyze_pose(pose_keypoints):
    """
    Analyze detected pose for activity recognition.
    
    Parameters:
    - pose_keypoints: Detected pose keypoints
    
    Returns:
    - Dictionary with pose analysis results
    """
```

#### Keypoint Structure
```python
pose_keypoints = {
    "nose": (x, y, confidence),
    "left_eye": (x, y, confidence),
    "right_eye": (x, y, confidence),
    "left_ear": (x, y, confidence),
    "right_ear": (x, y, confidence),
    "left_shoulder": (x, y, confidence),
    "right_shoulder": (x, y, confidence),
    "left_elbow": (x, y, confidence),
    "right_elbow": (x, y, confidence),
    "left_wrist": (x, y, confidence),
    "right_wrist": (x, y, confidence),
    "left_hip": (x, y, confidence),
    "right_hip": (x, y, confidence),
    "left_knee": (x, y, confidence),
    "right_knee": (x, y, confidence),
    "left_ankle": (x, y, confidence),
    "right_ankle": (x, y, confidence)
}
```

#### Applications
- **Activity Recognition**: Recognize human activities
- **Gesture Control**: Control systems with gestures
- **Fitness Tracking**: Track exercise and movement
- **Security**: Detect suspicious behavior patterns

### Mask Detection (`detection/mask_detection.py`)

#### Purpose
Detect whether individuals are wearing face masks for health and safety compliance.

#### Features
- **Mask Classification**: Classify mask vs. no mask
- **Mask Type Detection**: Detect different types of masks
- **Compliance Monitoring**: Monitor mask compliance
- **Real-time Detection**: Real-time mask detection

#### Detection Methods
1. **Custom CNN**: Custom convolutional neural network
2. **Transfer Learning**: Pre-trained models fine-tuned for masks
3. **Ensemble Methods**: Multiple models for improved accuracy
4. **Rule-based**: Rule-based detection using face landmarks

#### Key Functions
```python
def detect_mask(face_region):
    """
    Detect if a face is wearing a mask.
    
    Parameters:
    - face_region: Cropped face image
    
    Returns:
    - Dictionary with mask detection results
    """

def classify_mask_type(face_region):
    """
    Classify the type of mask being worn.
    
    Parameters:
    - face_region: Cropped face image
    
    Returns:
    - Mask type classification
    """
```

#### Mask Types
- **No Mask**: No face covering
- **Surgical Mask**: Medical/surgical masks
- **N95 Mask**: N95 respirator masks
- **Cloth Mask**: Cloth face coverings
- **Other**: Other types of face coverings

#### Applications
- **Health Compliance**: Monitor mask compliance
- **Safety Protocols**: Enforce safety protocols
- **Public Health**: Track public health measures
- **Access Control**: Control access based on mask compliance

### Lighting Preprocessing (`detection/lighting_preprocessing.py`)

#### Purpose
Enhance images for better detection and recognition under challenging lighting conditions.

#### Features
- **Histogram Equalization**: Improve image contrast
- **Adaptive Thresholding**: Adaptive thresholding for different lighting
- **Noise Reduction**: Reduce image noise
- **Color Correction**: Correct color balance

#### Preprocessing Methods
1. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
2. **Gamma Correction**: Adjust image gamma
3. **White Balance**: Correct white balance
4. **Denoising**: Remove image noise

#### Key Functions
```python
def enhance_lighting(image):
    """
    Enhance image lighting conditions.
    
    Parameters:
    - image: Input image
    
    Returns:
    - Enhanced image
    """

def normalize_lighting(image):
    """
    Normalize lighting across the image.
    
    Parameters:
    - image: Input image
    
    Returns:
    - Normalized image
    """
```

#### Enhancement Techniques
- **CLAHE**: Improves local contrast
- **Gamma Correction**: Adjusts brightness and contrast
- **Histogram Stretching**: Stretches histogram for better contrast
- **Adaptive Enhancement**: Adapts to local image characteristics

## Analysis Modules (`analysis/`)

### Body Shape Analysis (`analysis/body_shape_analysis.py`)

#### Purpose
Analyze human body shape and proportions for various applications including fitness tracking and body measurement.

#### Features
- **Body Segmentation**: Segment human body from background
- **Shape Analysis**: Analyze body shape and proportions
- **Measurement Estimation**: Estimate body measurements
- **Posture Analysis**: Analyze body posture

#### Analysis Methods
1. **Semantic Segmentation**: Segment body parts
2. **Keypoint Analysis**: Use pose keypoints for analysis
3. **Contour Analysis**: Analyze body contours
4. **Machine Learning**: ML-based body analysis

#### Key Functions
```python
def analyze_body_shape(frame, pose_keypoints):
    """
    Analyze body shape from pose keypoints.
    
    Parameters:
    - frame: Input image frame
    - pose_keypoints: Detected pose keypoints
    
    Returns:
    - Dictionary with body shape analysis
    """

def estimate_measurements(pose_keypoints):
    """
    Estimate body measurements from pose keypoints.
    
    Parameters:
    - pose_keypoints: Detected pose keypoints
    
    Returns:
    - Dictionary with estimated measurements
    """
```

#### Body Measurements
- **Height**: Estimated body height
- **Shoulder Width**: Shoulder width measurement
- **Waist**: Waist circumference
- **Hip**: Hip circumference
- **Body Mass Index**: Estimated BMI

#### Applications
- **Fitness Tracking**: Track fitness progress
- **Body Measurement**: Estimate body measurements
- **Health Monitoring**: Monitor health indicators
- **Fashion**: Virtual fitting and sizing

### Movement Analysis (`analysis/movement_analysis.py`)

#### Purpose
Analyze human movement patterns for activity recognition and behavior analysis.

#### Features
- **Motion Tracking**: Track human movement
- **Activity Recognition**: Recognize human activities
- **Gesture Analysis**: Analyze hand gestures
- **Behavior Analysis**: Analyze behavior patterns

#### Analysis Methods
1. **Optical Flow**: Track pixel movement
2. **Keypoint Tracking**: Track pose keypoints over time
3. **Deep Learning**: ML-based activity recognition
4. **Rule-based**: Rule-based activity recognition

#### Key Functions
```python
def analyze_movement(frame_sequence):
    """
    Analyze movement in a sequence of frames.
    
    Parameters:
    - frame_sequence: Sequence of frames
    
    Returns:
    - Dictionary with movement analysis
    """

def recognize_activity(movement_data):
    """
    Recognize human activity from movement data.
    
    Parameters:
    - movement_data: Movement analysis data
    
    Returns:
    - Activity classification
    """
```

#### Activity Types
- **Walking**: Normal walking motion
- **Running**: Running motion
- **Sitting**: Sitting down
- **Standing**: Standing up
- **Waving**: Hand waving gesture
- **Pointing**: Pointing gesture

#### Applications
- **Activity Recognition**: Recognize human activities
- **Gesture Control**: Control systems with gestures
- **Behavior Analysis**: Analyze behavior patterns
- **Security**: Detect suspicious behavior

### Clothes Detection (`analysis/clothes_detection.py`)

#### Purpose
Detect and classify clothing items for fashion analysis and security applications.

#### Features
- **Clothing Detection**: Detect clothing items
- **Clothing Classification**: Classify clothing types
- **Color Analysis**: Analyze clothing colors
- **Style Analysis**: Analyze clothing style

#### Detection Methods
1. **Object Detection**: Detect clothing as objects
2. **Semantic Segmentation**: Segment clothing regions
3. **Deep Learning**: ML-based clothing detection
4. **Color Analysis**: Analyze clothing colors

#### Key Functions
```python
def detect_clothes(frame):
    """
    Detect clothing items in a frame.
    
    Parameters:
    - frame: Input image frame
    
    Returns:
    - List of detected clothing items
    """

def classify_clothes(clothing_region):
    """
    Classify detected clothing items.
    
    Parameters:
    - clothing_region: Cropped clothing image
    
    Returns:
    - Clothing classification
    """
```

#### Clothing Categories
- **Tops**: Shirts, blouses, t-shirts
- **Bottoms**: Pants, skirts, shorts
- **Outerwear**: Jackets, coats, sweaters
- **Accessories**: Hats, bags, jewelry
- **Footwear**: Shoes, boots, sandals

#### Applications
- **Fashion Analysis**: Analyze fashion trends
- **Virtual Fitting**: Virtual clothing fitting
- **Security**: Identify individuals by clothing
- **Retail**: Analyze customer preferences

### Cooldown Manager (`analysis/cooldown_manager.py`)

#### Purpose
Manage cooldown periods and session memory to prevent duplicate processing and improve system efficiency.

#### Features
- **Session Memory**: Remember recent detections
- **Cooldown Management**: Manage cooldown periods
- **Duplicate Prevention**: Prevent duplicate processing
- **Performance Optimization**: Optimize system performance

#### Key Functions
```python
def should_process(user_id, cooldown_period):
    """
    Check if a user should be processed based on cooldown.
    
    Parameters:
    - user_id: User identifier
    - cooldown_period: Cooldown period in seconds
    
    Returns:
    - Boolean indicating if processing should occur
    """

def update_session(user_id, timestamp):
    """
    Update session memory with user activity.
    
    Parameters:
    - user_id: User identifier
    - timestamp: Activity timestamp
    """
```

#### Session Management
- **User Tracking**: Track user activity
- **Time-based Cooldowns**: Time-based cooldown management
- **Event-based Cooldowns**: Event-based cooldown management
- **Memory Management**: Efficient memory usage

#### Applications
- **Performance Optimization**: Reduce redundant processing
- **Resource Management**: Manage system resources
- **User Experience**: Improve user experience
- **System Efficiency**: Improve overall system efficiency

## Integration with AI Engine

### Automatic Integration
The detection and analysis modules integrate automatically with the AI Engine through the hook system:

```python
# In learning_hooks.py
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

### Event-Driven Processing
- **Face Detection Events**: Trigger analysis when faces are detected
- **Pose Detection Events**: Trigger analysis when poses are detected
- **Movement Events**: Trigger analysis when movement is detected
- **Custom Events**: Trigger custom analysis modules

## Performance Considerations

### Optimization Strategies
1. **Multi-threading**: Use multiple threads for parallel processing
2. **GPU Acceleration**: Utilize GPU for deep learning models
3. **Model Optimization**: Optimize models for real-time processing
4. **Caching**: Cache frequently used data

### Resource Management
- **Memory Management**: Efficient memory usage
- **CPU Usage**: Optimize CPU usage
- **GPU Usage**: Optimize GPU usage
- **Storage**: Efficient storage management

## Security and Privacy

### Privacy Considerations
- **Data Minimization**: Collect only necessary data
- **Anonymization**: Anonymize personal data
- **Consent**: Obtain proper consent
- **Retention**: Implement data retention policies

### Security Measures
- **Access Control**: Control access to analysis data
- **Encryption**: Encrypt sensitive data
- **Audit Logging**: Log all analysis activities
- **Secure Storage**: Secure storage of analysis data

## Troubleshooting

### Common Issues

#### Detection Accuracy
- **Problem**: Low detection accuracy
- **Solution**: Improve image quality, adjust parameters, retrain models

#### Performance Issues
- **Problem**: Slow processing
- **Solution**: Optimize algorithms, use GPU acceleration, reduce resolution

#### Memory Issues
- **Problem**: High memory usage
- **Solution**: Implement memory limits, optimize data structures

### Debug Tools
- **Logging**: Comprehensive logging for debugging
- **Metrics**: Performance metrics and monitoring
- **Visualization**: Visual debugging tools
- **Testing**: Unit tests and integration tests

## Best Practices

### Development
1. **Modular Design**: Keep modules independent and modular
2. **Error Handling**: Implement comprehensive error handling
3. **Testing**: Write comprehensive tests
4. **Documentation**: Maintain up-to-date documentation

### Performance
1. **Optimization**: Continuously optimize performance
2. **Monitoring**: Monitor system performance
3. **Profiling**: Profile code for bottlenecks
4. **Scaling**: Design for scalability

### Security
1. **Privacy**: Protect user privacy
2. **Security**: Implement security measures
3. **Compliance**: Ensure regulatory compliance
4. **Audit**: Regular security audits

---

This documentation provides comprehensive coverage of the Detection and Analysis modules, their functionality, and integration patterns. These modules extend the basic face recognition capabilities with advanced computer vision features for comprehensive human analysis and monitoring.
