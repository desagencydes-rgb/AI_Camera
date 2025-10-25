# Troubleshooting Guide

## Overview
This comprehensive troubleshooting guide provides solutions for common issues encountered when using the OCCUR-CALL AI Camera System. It covers installation problems, runtime errors, performance issues, and system configuration problems.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Camera and Hardware Issues](#camera-and-hardware-issues)
3. [Face Recognition Issues](#face-recognition-issues)
4. [Database Issues](#database-issues)
5. [Performance Issues](#performance-issues)
6. [Configuration Issues](#configuration-issues)
7. [Network and Connectivity Issues](#network-and-connectivity-issues)
8. [Security Issues](#security-issues)
9. [Debugging Tools](#debugging-tools)
10. [Common Error Messages](#common-error-messages)

## Installation Issues

### Python Environment Issues

#### Issue: Python Version Compatibility
**Error**: `Python 3.8 or higher is required`
**Solution**:
```bash
# Check Python version
python --version

# If version is too old, install Python 3.8+
# Download from https://www.python.org/downloads/
# Or use package manager:
# Windows: choco install python
# macOS: brew install python
# Linux: sudo apt install python3.8
```

#### Issue: Virtual Environment Problems
**Error**: `ModuleNotFoundError: No module named 'cv2'`
**Solution**:
```bash
# Create new virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Issue: Package Installation Failures
**Error**: `Failed building wheel for dlib`
**Solution**:
```bash
# Install build tools first
# Windows:
pip install cmake
pip install dlib
pip install face_recognition

# macOS:
brew install cmake
pip install dlib
pip install face_recognition

# Linux:
sudo apt install cmake
sudo apt install libopenblas-dev liblapack-dev
pip install dlib
pip install face_recognition
```

### Dependency Issues

#### Issue: OpenCV Installation Problems
**Error**: `ImportError: No module named 'cv2'`
**Solution**:
```bash
# Uninstall and reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python

# If still having issues, try:
pip install opencv-python-headless
```

#### Issue: Face Recognition Library Issues
**Error**: `ImportError: No module named 'face_recognition'`
**Solution**:
```bash
# Install face_recognition with dependencies
pip install cmake
pip install dlib
pip install face_recognition

# If dlib fails, try:
pip install --upgrade pip
pip install cmake
pip install dlib
pip install face_recognition
```

#### Issue: PyTorch Installation Problems
**Error**: `ImportError: No module named 'torch'`
**Solution**:
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (if available):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Camera and Hardware Issues

### Camera Access Problems

#### Issue: Camera Not Detected
**Error**: `[FATAL] Could not access camera. Check connection.`
**Solutions**:
1. **Check Camera Connection**:
   ```bash
   # Test camera with other applications
   # Try different USB ports
   # Check camera drivers
   ```

2. **Try Different Camera Index**:
   ```python
   # In camera_main.py, change:
   cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
   
   # Or test all available cameras:
   for i in range(10):
       cap = cv2.VideoCapture(i)
       if cap.isOpened():
           print(f"Camera {i} is available")
           cap.release()
   ```

3. **Check Camera Permissions**:
   ```bash
   # Windows: Check camera privacy settings
   # macOS: Check System Preferences > Security & Privacy > Camera
   # Linux: Check user permissions for /dev/video*
   ```

#### Issue: Camera Already in Use
**Error**: `Camera is already being used by another application`
**Solution**:
```bash
# Close other applications using the camera
# Check for:
# - Other camera applications
# - Video conferencing software
# - Browser tabs with camera access
# - Security software with camera access
```

#### Issue: Poor Camera Quality
**Symptoms**: Blurry images, low resolution, poor lighting
**Solutions**:
1. **Adjust Camera Settings**:
   ```python
   cap = cv2.VideoCapture(0)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
   cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
   cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
   ```

2. **Improve Lighting**:
   - Use natural lighting when possible
   - Avoid backlighting
   - Use diffused lighting
   - Ensure even lighting across the face

3. **Camera Positioning**:
   - Position camera at eye level
   - Maintain 2-3 feet distance
   - Ensure face is centered in frame
   - Avoid extreme angles

### Hardware Performance Issues

#### Issue: High CPU Usage
**Symptoms**: System slowdown, high CPU usage
**Solutions**:
1. **Reduce Processing Frequency**:
   ```python
   # Process every 3rd frame instead of every frame
   ai = AICameraEngine(every_n_frames=3)
   ```

2. **Use Detection-Only Mode**:
   ```python
   # Disable recognition for testing
   ai = AICameraEngine(recognizer="none")
   ```

3. **Reduce Image Resolution**:
   ```python
   cap = cv2.VideoCapture(0)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

#### Issue: Memory Issues
**Error**: `MemoryError` or system becomes unresponsive
**Solutions**:
1. **Reduce Face Database Size**:
   ```bash
   # Remove old face images
   # Keep only high-quality samples
   # Use fewer samples per user
   ```

2. **Use LBPH Backend**:
   ```python
   # LBPH uses less memory than face_recognition
   ai = AICameraEngine(recognizer="lbph")
   ```

3. **Implement Memory Limits**:
   ```python
   # Add memory monitoring
   import psutil
   
   def check_memory():
       memory_percent = psutil.virtual_memory().percent
       if memory_percent > 80:
           print("High memory usage detected")
   ```

## Face Recognition Issues

### Recognition Backend Problems

#### Issue: Face Recognition Not Working
**Error**: `[AI] Recognition backend: NONE (detection-only)`
**Solutions**:
1. **Check Face Database**:
   ```bash
   # Ensure face database exists
   ls data/face_db/
   
   # Check if user directories exist
   ls data/face_db/user_*/
   
   # Verify face images are present
   ls data/face_db/user_*/sample_*.jpg
   ```

2. **Verify Face Recognition Installation**:
   ```python
   # Test face_recognition import
   try:
       import face_recognition
       print("face_recognition is available")
   except ImportError:
       print("face_recognition not installed")
   ```

3. **Check Image Formats**:
   ```bash
   # Ensure images are in supported formats
   # Supported: .jpg, .jpeg, .png, .bmp
   # Check file sizes (not too large/small)
   ```

#### Issue: Low Recognition Accuracy
**Symptoms**: Faces not recognized, false positives
**Solutions**:
1. **Improve Image Quality**:
   - Use high-resolution images
   - Ensure good lighting
   - Use multiple angles
   - Avoid blurry images

2. **Add More Training Samples**:
   ```python
   # Register multiple samples per user
   for image_path in user_images:
       image = cv2.imread(image_path)
       ai.register_user("user_id", image)
   ```

3. **Adjust Recognition Parameters**:
   ```python
   # In ai_engine.py, adjust confidence threshold
   if dist <= 0.6:  # Try 0.5 or 0.7
       return self.known_labels[idx], dist
   ```

#### Issue: False Positives
**Symptoms**: Wrong person recognized
**Solutions**:
1. **Increase Confidence Threshold**:
   ```python
   # Increase threshold for stricter matching
   if dist <= 0.5:  # Instead of 0.6
       return self.known_labels[idx], dist
   ```

2. **Improve Training Data**:
   - Use diverse lighting conditions
   - Include different expressions
   - Use high-quality images
   - Remove low-quality samples

3. **Use Multiple Recognition Backends**:
   ```python
   # Implement ensemble recognition
   def ensemble_recognition(face_image):
       # Try multiple backends
       results = []
       # Combine results for better accuracy
   ```

### Face Database Issues

#### Issue: Face Database Not Loading
**Error**: `Failed to load face database`
**Solutions**:
1. **Check Directory Structure**:
   ```bash
   # Ensure proper directory structure
   data/
   └── face_db/
       ├── user_001/
       │   ├── sample_001.jpg
       │   └── sample_002.jpg
       └── user_002/
           └── sample_001.jpg
   ```

2. **Verify File Permissions**:
   ```bash
   # Check file permissions
   ls -la data/face_db/
   
   # Fix permissions if needed
   chmod -R 755 data/face_db/
   ```

3. **Check Image Validity**:
   ```python
   # Test image loading
   import cv2
   
   for image_path in face_images:
       image = cv2.imread(image_path)
       if image is None:
           print(f"Invalid image: {image_path}")
   ```

#### Issue: Face Database Corruption
**Symptoms**: Recognition fails, database errors
**Solutions**:
1. **Backup and Restore**:
   ```bash
   # Backup current database
   cp -r data/face_db data/face_db_backup
   
   # Restore from backup
   cp -r data/face_db_backup data/face_db
   ```

2. **Rebuild Database**:
   ```python
   # Clear and rebuild database
   import shutil
   shutil.rmtree("data/face_db")
   os.makedirs("data/face_db")
   
   # Re-register users
   for user_id, images in user_data.items():
       for image_path in images:
           image = cv2.imread(image_path)
           ai.register_user(user_id, image)
   ```

## Database Issues

### SQLite Database Problems

#### Issue: Database Connection Errors
**Error**: `sqlite3.OperationalError: unable to open database file`
**Solutions**:
1. **Check File Permissions**:
   ```bash
   # Check database file permissions
   ls -la Update_and_Backup/face_events.db
   
   # Fix permissions
   chmod 664 Update_and_Backup/face_events.db
   ```

2. **Check Disk Space**:
   ```bash
   # Check available disk space
   df -h
   
   # Clean up if needed
   rm -rf data/snapshots/old_*
   ```

3. **Check Directory Existence**:
   ```bash
   # Ensure directory exists
   mkdir -p Update_and_Backup
   ```

#### Issue: Database Corruption
**Error**: `sqlite3.DatabaseError: database disk image is malformed`
**Solutions**:
1. **Check Database Integrity**:
   ```python
   import sqlite3
   
   conn = sqlite3.connect("Update_and_Backup/face_events.db")
   cursor = conn.cursor()
   cursor.execute("PRAGMA integrity_check")
   result = cursor.fetchone()
   print(f"Database integrity: {result}")
   conn.close()
   ```

2. **Restore from Backup**:
   ```bash
   # Restore from backup
   cp backups/face_events_backup.db Update_and_Backup/face_events.db
   ```

3. **Recreate Database**:
   ```python
   # Recreate database
   import os
   os.remove("Update_and_Backup/face_events.db")
   
   # Run initialization
   from Update_and_Backup.update_db import initialize_db
   initialize_db()
   ```

#### Issue: Database Performance Issues
**Symptoms**: Slow database operations, timeouts
**Solutions**:
1. **Optimize Database**:
   ```python
   import sqlite3
   
   conn = sqlite3.connect("Update_and_Backup/face_events.db")
   cursor = conn.cursor()
   
   # Create indexes
   cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON face_events(timestamp)")
   cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON face_events(user_id)")
   
   # Optimize database
   cursor.execute("VACUUM")
   conn.close()
   ```

2. **Implement Connection Pooling**:
   ```python
   # Use connection pooling
   from sqlite3 import connect
   import threading
   
   class DatabasePool:
       def __init__(self, db_path, max_connections=5):
           self.db_path = db_path
           self.max_connections = max_connections
           self.connections = []
           self.lock = threading.Lock()
   ```

## Performance Issues

### System Performance

#### Issue: Slow Face Detection
**Symptoms**: Delayed face detection, low FPS
**Solutions**:
1. **Optimize Detection Parameters**:
   ```python
   # Adjust detection parameters
   rects = self.face_cascade.detectMultiScale(
       gray, 
       scaleFactor=1.3,  # Increase for speed
       minNeighbors=3,    # Decrease for speed
       minSize=(50, 50)   # Increase for speed
   )
   ```

2. **Use GPU Acceleration**:
   ```python
   # Use GPU for face recognition
   import face_recognition
   
   # Enable GPU acceleration if available
   face_locations = face_recognition.face_locations(
       image, 
       model="hog"  # Use HOG for CPU, CNN for GPU
   )
   ```

3. **Implement Frame Skipping**:
   ```python
   # Process every N frames
   ai = AICameraEngine(every_n_frames=2)
   ```

#### Issue: High Memory Usage
**Symptoms**: System slowdown, memory errors
**Solutions**:
1. **Implement Memory Monitoring**:
   ```python
   import psutil
   
   def monitor_memory():
       memory = psutil.virtual_memory()
       if memory.percent > 80:
           print("High memory usage detected")
           # Implement cleanup
   ```

2. **Optimize Data Structures**:
   ```python
   # Use efficient data structures
   import numpy as np
   
   # Use numpy arrays instead of lists
   features = np.array(feature_list)
   
   # Use appropriate data types
   features = features.astype(np.float32)
   ```

3. **Implement Cleanup**:
   ```python
   # Regular cleanup
   def cleanup_old_data():
       # Remove old snapshots
       # Clear unused variables
       # Garbage collect
       import gc
       gc.collect()
   ```

### Network Performance

#### Issue: Slow Network Operations
**Symptoms**: Delayed responses, timeouts
**Solutions**:
1. **Implement Timeouts**:
   ```python
   import requests
   
   # Set timeouts
   response = requests.get(url, timeout=5)
   ```

2. **Use Async Operations**:
   ```python
   import asyncio
   
   async def async_operation():
       # Perform async operations
       pass
   ```

3. **Implement Caching**:
   ```python
   # Cache frequently accessed data
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def cached_function(param):
       return expensive_operation(param)
   ```

## Configuration Issues

### Configuration File Problems

#### Issue: Configuration Not Loading
**Error**: `Configuration file not found`
**Solutions**:
1. **Check File Path**:
   ```python
   # Verify config file exists
   import os
   config_path = "config.py"
   if not os.path.exists(config_path):
       print("Config file not found")
   ```

2. **Check File Permissions**:
   ```bash
   # Check file permissions
   ls -la config.py
   
   # Fix permissions
   chmod 644 config.py
   ```

3. **Validate Configuration**:
   ```python
   # Validate configuration values
   def validate_config():
       required_dirs = [DATA_DIR, FACE_DB_DIR, SNAPSHOT_DIR]
       for dir_path in required_dirs:
           if not dir_path.exists():
               print(f"Directory not found: {dir_path}")
   ```

#### Issue: Invalid Configuration Values
**Error**: `Invalid configuration value`
**Solutions**:
1. **Check Configuration Values**:
   ```python
   # Validate configuration
   def validate_config():
       if not isinstance(every_n_frames, int) or every_n_frames < 1:
           raise ValueError("every_n_frames must be positive integer")
   ```

2. **Provide Default Values**:
   ```python
   # Use default values
   every_n_frames = config.get('every_n_frames', 1)
   unknown_cooldown = config.get('unknown_cooldown', 15.0)
   ```

### Environment Variables

#### Issue: Environment Variables Not Set
**Error**: `Environment variable not set`
**Solutions**:
1. **Set Environment Variables**:
   ```bash
   # Set environment variables
   export AI_CAMERA_DB_PATH="/path/to/database"
   export AI_CAMERA_FACE_DB="/path/to/face_db"
   ```

2. **Use Default Values**:
   ```python
   import os
   
   # Use environment variables with defaults
   db_path = os.getenv('AI_CAMERA_DB_PATH', 'default/path')
   face_db = os.getenv('AI_CAMERA_FACE_DB', 'default/face_db')
   ```

## Network and Connectivity Issues

### Network Configuration

#### Issue: Network Connection Problems
**Error**: `Connection refused` or `Timeout`
**Solutions**:
1. **Check Network Connectivity**:
   ```bash
   # Test network connectivity
   ping google.com
   
   # Check DNS resolution
   nslookup google.com
   ```

2. **Configure Proxy**:
   ```python
   # Configure proxy if needed
   import requests
   
   proxies = {
       'http': 'http://proxy:port',
       'https': 'https://proxy:port'
   }
   response = requests.get(url, proxies=proxies)
   ```

3. **Implement Retry Logic**:
   ```python
   import time
   
   def retry_operation(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise e
               time.sleep(2 ** attempt)  # Exponential backoff
   ```

### API Connectivity

#### Issue: API Connection Failures
**Error**: `API connection failed`
**Solutions**:
1. **Check API Endpoints**:
   ```python
   # Test API endpoints
   import requests
   
   def test_api_endpoint(url):
       try:
           response = requests.get(url, timeout=5)
           return response.status_code == 200
       except Exception as e:
           print(f"API test failed: {e}")
           return False
   ```

2. **Implement Health Checks**:
   ```python
   # Implement health checks
   def health_check():
       checks = [
           check_database_connection(),
           check_camera_connection(),
           check_face_recognition()
       ]
       return all(checks)
   ```

## Security Issues

### Authentication Problems

#### Issue: Authentication Failures
**Error**: `Authentication failed`
**Solutions**:
1. **Check Credentials**:
   ```python
   # Validate credentials
   def validate_credentials(username, password):
       # Implement proper validation
       return username == "admin" and password == "secret"
   ```

2. **Implement Token-Based Authentication**:
   ```python
   import jwt
   
   def generate_token(user_id):
       payload = {'user_id': user_id}
       token = jwt.encode(payload, secret_key, algorithm='HS256')
       return token
   ```

### Access Control Issues

#### Issue: Unauthorized Access
**Error**: `Access denied`
**Solutions**:
1. **Implement Role-Based Access Control**:
   ```python
   def check_permissions(user_id, resource, action):
       user_role = get_user_role(user_id)
       required_permission = f"{resource}:{action}"
       return required_permission in get_role_permissions(user_role)
   ```

2. **Implement Audit Logging**:
   ```python
   def log_access_attempt(user_id, resource, action, success):
       log_entry = {
           'timestamp': datetime.now(),
           'user_id': user_id,
           'resource': resource,
           'action': action,
           'success': success
       }
       log_to_database(log_entry)
   ```

## Debugging Tools

### Logging and Monitoring

#### Enable Debug Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Use in code
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Performance Monitoring
```python
import time
import psutil

def monitor_performance():
    # CPU usage
    cpu_percent = psutil.cpu_percent()
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    print(f"CPU: {cpu_percent}%")
    print(f"Memory: {memory.percent}%")
    print(f"Disk: {disk.percent}%")
```

### Testing Tools

#### Unit Testing
```python
import unittest

class TestAICameraEngine(unittest.TestCase):
    def setUp(self):
        self.ai = AICameraEngine(recognizer="none")
    
    def test_face_detection(self):
        # Test face detection
        frame = cv2.imread("test_image.jpg")
        annotated, faces = self.ai.detect_and_recognize(frame)
        self.assertIsInstance(faces, list)
    
    def test_user_registration(self):
        # Test user registration
        image = cv2.imread("test_face.jpg")
        path = self.ai.register_user("test_user", image)
        self.assertTrue(path.exists())

if __name__ == '__main__':
    unittest.main()
```

#### Integration Testing
```python
def test_system_integration():
    """Test complete system integration."""
    # Initialize system
    ai = AICameraEngine()
    cap = cv2.VideoCapture(0)
    
    # Test face detection
    ret, frame = cap.read()
    if ret:
        annotated, faces = ai.detect_and_recognize(frame)
        assert len(faces) >= 0
    
    # Test database logging
    from Utils.db_utils import log_face_event
    log_face_event("test", "test_user", confidence=0.95)
    
    # Test learning system
    from learning.learning_core import get_learning_core
    learning_core = get_learning_core()
    
    cap.release()
    print("Integration test passed")
```

## Common Error Messages

### System Errors

#### `[FATAL] Failed to initialize AI Engine`
**Causes**: Missing dependencies, configuration errors, hardware issues
**Solutions**:
1. Check all dependencies are installed
2. Verify configuration files
3. Check camera hardware
4. Review error logs

#### `[ERROR] Failed to grab frame`
**Causes**: Camera access issues, hardware problems
**Solutions**:
1. Check camera connection
2. Close other applications using camera
3. Check camera drivers
4. Try different camera index

#### `[AI ERROR] Processing frame failed`
**Causes**: Memory issues, corrupted data, system overload
**Solutions**:
1. Check system memory
2. Reduce processing load
3. Restart system
4. Check for corrupted data

### Database Errors

#### `[DB ERROR] Failed to log face event`
**Causes**: Database connection issues, permission problems
**Solutions**:
1. Check database file permissions
2. Verify disk space
3. Check database integrity
4. Restart database service

#### `sqlite3.OperationalError: database is locked`
**Causes**: Multiple processes accessing database, improper cleanup
**Solutions**:
1. Ensure proper connection cleanup
2. Use connection pooling
3. Check for hanging processes
4. Restart system if necessary

### Recognition Errors

#### `[AI] Recognition backend: NONE (detection-only)`
**Causes**: Missing face_recognition library, empty face database
**Solutions**:
1. Install face_recognition library
2. Check face database structure
3. Add face images to database
4. Verify image formats

#### `RecognitionError: No faces found in image`
**Causes**: Poor image quality, incorrect face detection
**Solutions**:
1. Improve image quality
2. Check lighting conditions
3. Adjust detection parameters
4. Use different face detection method

## Best Practices for Troubleshooting

### Systematic Approach
1. **Identify the Problem**: Clearly define what's not working
2. **Gather Information**: Collect error messages, logs, and system state
3. **Check Common Causes**: Review common issues and solutions
4. **Test Solutions**: Try solutions one at a time
5. **Document Results**: Record what worked and what didn't

### Prevention
1. **Regular Maintenance**: Keep system updated and maintained
2. **Monitoring**: Implement monitoring and alerting
3. **Backups**: Regular backups of data and configuration
4. **Testing**: Regular testing of system components
5. **Documentation**: Keep documentation up to date

### Getting Help
1. **Check Logs**: Review system logs for error details
2. **Search Documentation**: Look for similar issues in documentation
3. **Test Components**: Test individual components in isolation
4. **Contact Support**: Contact technical support if needed
5. **Community Forums**: Use community forums for help

---

This troubleshooting guide provides comprehensive solutions for common issues in the OCCUR-CALL AI Camera System. For additional help, refer to the individual module documentation and the main README file.
