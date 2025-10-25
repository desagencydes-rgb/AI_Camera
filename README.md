# OCCUR-CALL AI Camera System

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Installation Guide](#installation-guide)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Module Documentation](#module-documentation)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Overview

The OCCUR-CALL AI Camera System is a comprehensive real-time face recognition and analysis platform designed for security, attendance tracking, and intelligent monitoring applications. The system combines multiple AI engines to provide robust face detection, recognition, body analysis, and learning capabilities.

### Key Capabilities
- **Real-time Face Detection & Recognition**: Multi-backend support with face_recognition and OpenCV LBPH
- **Intelligent Learning System**: Continuous learning from detected faces with embedding storage
- **Multi-Engine Architecture**: Modular design supporting face, body, object, and authentication engines
- **Database Integration**: SQLite-based logging and storage with backup capabilities
- **Enhanced Recognition**: Advanced preprocessing for challenging lighting and pose conditions
- **Hook System**: Extensible architecture for custom analysis modules

## Features

### Core Features
- ✅ Real-time face detection using Haar cascades
- ✅ Multi-backend face recognition (face_recognition, OpenCV LBPH)
- ✅ Automatic unknown face capture and storage
- ✅ Database logging of all face events
- ✅ Learning system with embedding storage
- ✅ Enhanced recognition for challenging conditions
- ✅ Multi-camera support
- ✅ Session memory and cooldown management

### Advanced Features
- ✅ Body shape analysis
- ✅ Movement analysis
- ✅ Clothes detection
- ✅ Eye detection
- ✅ Pose detection
- ✅ Mask detection
- ✅ Lighting preprocessing
- ✅ Incremental learning
- ✅ Model management
- ✅ Backup and update systems

## System Architecture

```
AI_Camera/
├── camera_main.py              # Main application entry point
├── config.py                   # System configuration
├── requirements.txt            # Python dependencies
├── engines/                    # Core AI engines
│   ├── ai_engine.py           # Main face recognition engine
│   ├── ai_engine_enhancer.py  # Recognition enhancements
│   ├── auth_engine.py         # Authentication engine
│   ├── body_engine.py         # Body analysis engine
│   └── object_engine.py       # Object detection engine
├── detection/                  # Detection modules
│   ├── face_detection.py      # Face detection utilities
│   ├── eye_detection.py       # Eye detection
│   ├── pose_detection.py      # Pose estimation
│   ├── mask_detection.py      # Mask detection
│   └── lighting_preprocessing.py # Image enhancement
├── analysis/                   # Analysis modules
│   ├── body_shape_analysis.py # Body shape analysis
│   ├── movement_analysis.py   # Movement tracking
│   ├── clothes_detection.py   # Clothing detection
│   └── cooldown_manager.py    # Session management
├── learning/                   # Learning system
│   ├── learning_core.py       # Core learning functionality
│   ├── learning_hooks.py      # Hook system
│   ├── data_collector.py      # Data collection
│   ├── feature_analyzer.py    # Feature analysis
│   ├── incremental_trainer.py # Incremental learning
│   └── model_manager.py       # Model management
├── storage/                    # Storage utilities
│   ├── snapshot_manager.py    # Snapshot management
│   ├── logger_secure.py       # Secure logging
│   └── face_verification_storage.py # Face verification storage
├── Utils/                      # Utility modules
│   ├── db_utils.py            # Database utilities
│   └── storage_utils.py       # Storage utilities
├── data/                       # Data storage
│   ├── face_db/               # Face database
│   ├── snapshots/             # Captured snapshots
│   ├── unknown_faces/         # Unknown face captures
│   ├── learning_db/           # Learning database
│   └── models/                # AI models
└── Update_and_Backup/          # Backup and update system
    ├── update_db.py           # Database updates
    └── face_events.db         # Face events database
```

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10.0.26100)
- Webcam or camera device
- At least 4GB RAM (8GB recommended)
- 2GB free disk space

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd AI_Camera
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Or using PowerShell
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# If you encounter issues with dlib, try:
pip install cmake
pip install dlib
pip install face_recognition
```

### Step 4: Verify Installation
```bash
# Test basic functionality
python tests/test_ai_engine.py
```

### Step 5: Initialize Database
```bash
# Initialize the database
python Update_and_Backup/update_db.py
```

### Step 6: Run the System
```bash
# Start the AI Camera system
python camera_main.py
```

## Configuration

### Main Configuration (`config.py`)
The system configuration is managed through `config.py`:

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

### Camera Configuration (`camera_main.py`)
```python
# Enable/disable features
ENABLE_DB_LOGGING = True
USE_ENHANCER = True

# Directory paths
SNAPSHOT_DIR = r"C:\AI_Camera\data\snapshots"
UNKNOWN_DIR = r"C:\AI_Camera\data\unknown_faces"
```

### AI Engine Configuration
```python
# Initialize AI Engine with options
ai = AICameraEngine(
    recognizer="auto",           # "auto", "face_recognition", "lbph", "none"
    every_n_frames=1,           # Process every N frames
    draw=True,                  # Draw bounding boxes
    unknown_cooldown=15.0,      # Cooldown for unknown faces (seconds)
    snapshot_cooldown=5.0,      # Cooldown for snapshots (seconds)
    enable_db_logging=True      # Enable database logging
)
```

## Usage

### Basic Usage
1. **Start the System**: Run `python camera_main.py`
2. **Camera Window**: A window titled "OCCUR-CALL Camera" will open
3. **Face Detection**: Green boxes indicate recognized faces, red boxes indicate unknown faces
4. **Controls**:
   - `q` or `ESC`: Exit the application
   - `r`: Reload AI engine models

### Adding New Users
1. Place face images in `data/face_db/[username]/` directory
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
3. Press `r` to reload models, or restart the application

### Database Operations
```python
# Log face events manually
from Utils.db_utils import log_face_event, log_system_event

log_face_event(
    event_type="recognized",
    user_id="john_doe",
    confidence=0.95
)

log_system_event(
    event_type="system_start",
    details="AI Camera system started"
)
```

### Learning System Usage
```python
from learning.learning_core import get_learning_core

# Get learning core instance
learning_core = get_learning_core()

# Add features
learning_core.add_feature(
    label="user_123",
    feature_vector=embedding_vector,
    metadata={"timestamp": time.time()}
)

# Query similar features
results = learning_core.query_feature(embedding_vector, top_k=3)
```

### Hook System Usage
```python
from learning.learning_hooks import register_hook

def custom_analysis_hook(event_data):
    faces = event_data.get("faces", [])
    frame = event_data.get("frame")
    # Your custom analysis here
    print(f"Analyzing {len(faces)} faces")

# Register the hook
register_hook("custom_analysis", custom_analysis_hook, cooldown=2.0)
```

## Module Documentation

### Core Engines

#### AI Engine (`engines/ai_engine.py`)
The main face recognition engine providing:
- Multi-backend recognition (face_recognition, OpenCV LBPH)
- Face detection using Haar cascades
- Unknown face handling and auto-registration
- Embedding extraction for learning system
- Database integration

**Key Methods:**
- `detect_and_recognize(frame)`: Main processing method
- `register_user(user_id, image, bbox)`: Register new user
- `reload_models()`: Reload recognition models
- `_extract_face_embedding(frame, bbox)`: Extract face embeddings

#### AI Engine Enhancer (`engines/ai_engine_enhancer.py`)
Enhancement module providing:
- Lighting normalization
- Face rotation correction
- Scaling for distance variations
- Threaded recognition
- Session memory management

**Key Functions:**
- `enhance_engine(engine)`: Apply enhancements to AI engine
- `normalize_lighting(face_gray)`: Improve lighting conditions
- `rotate_face(face_img, angle)`: Rotate face images
- `scale_face(face_img, scale_factor)`: Scale face images

### Detection Modules

#### Face Detection (`detection/face_detection.py`)
Core face detection utilities using OpenCV Haar cascades.

#### Eye Detection (`detection/eye_detection.py`)
Eye detection and tracking capabilities.

#### Pose Detection (`detection/pose_detection.py`)
Human pose estimation and tracking.

#### Mask Detection (`detection/mask_detection.py`)
Face mask detection capabilities.

#### Lighting Preprocessing (`detection/lighting_preprocessing.py`)
Image enhancement for challenging lighting conditions.

### Analysis Modules

#### Body Shape Analysis (`analysis/body_shape_analysis.py`)
Body shape analysis and measurement capabilities.

#### Movement Analysis (`analysis/movement_analysis.py`)
Movement tracking and analysis.

#### Clothes Detection (`analysis/clothes_detection.py`)
Clothing detection and classification.

#### Cooldown Manager (`analysis/cooldown_manager.py`)
Session management and cooldown handling.

### Learning System

#### Learning Core (`learning/learning_core.py`)
Core learning functionality providing:
- Feature vector storage and retrieval
- Nearest neighbor queries
- Metadata management
- Persistent storage

**Key Methods:**
- `add_feature(label, feature_vector, metadata)`: Add new features
- `query_feature(feature_vector, top_k)`: Query similar features
- `save_db()`: Persist data to disk
- `get_metadata(label)`: Retrieve metadata

#### Learning Hooks (`learning/learning_hooks.py`)
Hook system for extensible analysis:
- Hook registration and management
- Async execution support
- Cooldown management
- Learning core integration

**Key Functions:**
- `register_hook(name, func, cooldown)`: Register analysis hooks
- `trigger_hooks(event_data, async_mode)`: Execute registered hooks
- `query_learning_system(embedding, top_k)`: Query learning system

### Storage Utilities

#### Database Utils (`Utils/db_utils.py`)
Database operations and logging:
- Face event logging
- System event logging
- Table management
- Error handling

#### Snapshot Manager (`storage/snapshot_manager.py`)
Snapshot capture and management utilities.

#### Secure Logger (`storage/logger_secure.py`)
Secure logging capabilities.

## Security Considerations

### Data Privacy
- **Face Data**: All face images are stored locally
- **Database**: SQLite databases contain sensitive information
- **Logs**: System logs may contain personal information

### Security Risks
1. **Unauthorized Access**: Face database could be accessed by unauthorized users
2. **Data Leakage**: Face images and recognition data could be compromised
3. **Privacy Violations**: Continuous monitoring may violate privacy laws
4. **Database Security**: SQLite databases are not encrypted by default

### Security Recommendations
1. **Access Control**: Implement proper file system permissions
2. **Encryption**: Consider encrypting sensitive data
3. **Network Security**: Ensure secure network connections
4. **Regular Updates**: Keep dependencies updated
5. **Audit Logging**: Monitor system access and usage
6. **Data Retention**: Implement data retention policies
7. **Compliance**: Ensure compliance with privacy regulations (GDPR, CCPA)

### Best Practices
- Use strong authentication for system access
- Regularly backup and secure database files
- Implement proper error handling to prevent information leakage
- Use HTTPS for any network communications
- Regular security audits and penetration testing
- Employee training on data privacy and security

## Troubleshooting

### Common Issues

#### Camera Not Detected
```
[FATAL] Could not access camera. Check connection.
```
**Solutions:**
- Check camera connection
- Ensure camera is not used by another application
- Try different camera index: `cv2.VideoCapture(1)`
- Update camera drivers

#### Face Recognition Not Working
```
[AI] Recognition backend: NONE (detection-only)
```
**Solutions:**
- Install face_recognition: `pip install face_recognition`
- Install dlib: `pip install dlib`
- Check face database directory exists
- Verify face images are in correct format

#### Database Errors
```
[DB ERROR] Failed to log face event
```
**Solutions:**
- Check database file permissions
- Ensure database directory exists
- Verify SQLite installation
- Check disk space

#### Performance Issues
**Solutions:**
- Reduce `every_n_frames` parameter
- Use smaller face database
- Enable threaded recognition
- Optimize image resolution

### Debug Mode
Enable debug logging by modifying `camera_main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### System Requirements Check
```python
# Check Python version
import sys
print(f"Python version: {sys.version}")

# Check OpenCV
import cv2
print(f"OpenCV version: {cv2.__version__}")

# Check face_recognition
try:
    import face_recognition
    print("face_recognition: Available")
except ImportError:
    print("face_recognition: Not available")
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add comprehensive docstrings
- Include error handling

### Testing
- Run existing tests: `python tests/test_ai_engine.py`
- Add new tests for new features
- Test on different camera configurations

### Documentation
- Update README for new features
- Add inline code comments
- Update module documentation
- Include usage examples

---

## License

This project is part of the OCCUR-CALL system. Please refer to the main project license for usage terms and conditions.

## Support

For technical support and questions:
- Check the troubleshooting section
- Review the module documentation
- Submit issues through the project repository
- Contact the development team

---

**Version**: 4.0  
**Last Updated**: 2024  
**Compatibility**: Python 3.8+, Windows 10/11, OpenCV 4.x
