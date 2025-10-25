# Storage and Database Documentation

## Overview
The Storage and Database system provides comprehensive data management capabilities for the OCCUR-CALL AI Camera System. It handles face data storage, event logging, backup operations, and secure data management.

## Architecture

### Storage Components

```
storage/
├── snapshot_manager.py           # Snapshot management
├── logger_secure.py              # Secure logging
└── face_verification_storage.py   # Face verification storage

Utils/
├── db_utils.py                   # Database utilities
└── storage_utils.py              # Storage utilities

Update_and_Backup/
├── update_db.py                  # Database updates
└── face_events.db               # Face events database

data/
├── face_db/                      # Face database
├── snapshots/                    # Captured snapshots
├── unknown_faces/                # Unknown face captures
├── learning_db/                  # Learning database
└── models/                       # AI models
```

## Database System

### Database Utilities (`Utils/db_utils.py`)

#### Purpose
The Database Utilities module provides a unified interface for logging face events and system events to SQLite databases.

#### Core Functions

##### log_face_event()
```python
def log_face_event(event_type: str, user_id: str = None, image_path: str = None, confidence: float = None) -> None
```
**Purpose**: Log face detection and recognition events to the database.

**Parameters**:
- `event_type`: Type of event ("recognized", "unknown", "auto_register")
- `user_id`: User identifier (optional)
- `image_path`: Path to saved image (optional)
- `confidence`: Recognition confidence score (optional)

**Process**:
1. Ensure face_events table exists
2. Connect to SQLite database
3. Insert event record with timestamp
4. Commit transaction and close connection
5. Handle errors gracefully

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

##### log_system_event()
```python
def log_system_event(event_type: str, details: str = None) -> None
```
**Purpose**: Log system events and status changes.

**Parameters**:
- `event_type`: Type of system event
- `details`: Additional event details (optional)

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

#### Database Schema

##### face_events Table
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

**Fields**:
- `id`: Primary key, auto-incrementing
- `timestamp`: ISO format timestamp
- `event_type`: Event type ("recognized", "unknown", "auto_register")
- `user_id`: User identifier
- `image_path`: Path to saved image
- `confidence`: Recognition confidence (0.0-1.0)

##### system_events Table
```sql
CREATE TABLE system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details TEXT
)
```

**Fields**:
- `id`: Primary key, auto-incrementing
- `timestamp`: ISO format timestamp
- `event_type`: System event type
- `details`: Additional event details

#### Error Handling
- **Database Connection Errors**: Handle connection failures gracefully
- **Transaction Errors**: Rollback failed transactions
- **File Permission Errors**: Handle permission issues
- **Disk Space Errors**: Handle disk space issues

### Database Update System (`Update_and_Backup/update_db.py`)

#### Purpose
The Database Update System handles database initialization, schema updates, and data migration.

#### Core Functions

##### initialize_db()
```python
def initialize_db():
    """
    Initialize the main OCCUR-CALL database with required tables.
    """
```

**Process**:
1. Connect to main OCCUR-CALL database
2. Create users table
3. Create attendance_logs table
4. Create unknown_faces table
5. Handle data migration from old tables
6. Commit changes and close connection

#### Database Schema

##### users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    role TEXT,
    face_encoding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Fields**:
- `id`: Primary key, auto-incrementing
- `name`: User's full name
- `email`: User's email address (unique)
- `role`: User role/permissions
- `face_encoding`: Stored face encoding (BLOB)
- `created_at`: Account creation timestamp

##### attendance_logs Table
```sql
CREATE TABLE attendance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    status TEXT CHECK(status IN ('recognized', 'manual', 'corrected')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    snapshot_path TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
```

**Fields**:
- `id`: Primary key, auto-incrementing
- `user_id`: Foreign key to users table
- `status`: Attendance status
- `timestamp`: Log timestamp
- `snapshot_path`: Path to attendance snapshot

##### unknown_faces Table
```sql
CREATE TABLE unknown_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_path TEXT,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP,
    seen_count INTEGER DEFAULT 1
)
```

**Fields**:
- `id`: Primary key, auto-incrementing
- `snapshot_path`: Path to unknown face snapshot
- `first_seen`: First detection timestamp
- `last_seen`: Last detection timestamp
- `seen_count`: Number of times detected

#### Data Migration
The system handles migration from old table structures:

```python
# Migration from old 'unknown' table
try:
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='unknown'")
    if c.fetchone():
        print("[DB] Migrating old 'unknown' table data...")
        c.execute("INSERT INTO unknown_faces (snapshot_path, first_seen, last_seen, seen_count) "
                  "SELECT snapshot_path, first_seen, last_seen, seen_count FROM unknown")
        c.execute("DROP TABLE unknown")
        print("[DB] Migration complete. Old 'unknown' table removed.")
except Exception as e:
    print(f"[DB] Migration check skipped: {e}")
```

## Storage Management

### Snapshot Manager (`storage/snapshot_manager.py`)

#### Purpose
The Snapshot Manager handles the capture, storage, and management of face snapshots.

#### Features
- **Automatic Capture**: Automatically capture face snapshots
- **Quality Control**: Ensure snapshot quality
- **Storage Organization**: Organize snapshots by user/date
- **Cleanup Management**: Manage old snapshots

#### Key Functions
```python
def capture_snapshot(frame, bbox, user_id=None):
    """
    Capture a face snapshot from a frame.
    
    Parameters:
    - frame: Input camera frame
    - bbox: Face bounding box (x, y, w, h)
    - user_id: User identifier (optional)
    
    Returns:
    - Path to saved snapshot
    """

def organize_snapshots():
    """
    Organize snapshots by user and date.
    """

def cleanup_old_snapshots(days_old=30):
    """
    Remove snapshots older than specified days.
    
    Parameters:
    - days_old: Age threshold in days
    """
```

#### Storage Structure
```
data/snapshots/
├── 2024/
│   ├── 01/
│   │   ├── 15/
│   │   │   ├── user_123/
│   │   │   │   ├── snapshot_001.jpg
│   │   │   │   └── snapshot_002.jpg
│   │   │   └── unknown/
│   │   │       ├── unknown_001.jpg
│   │   │       └── unknown_002.jpg
│   │   └── 16/
│   └── 02/
└── 2025/
```

### Face Verification Storage (`storage/face_verification_storage.py`)

#### Purpose
The Face Verification Storage module handles the storage and retrieval of face verification data.

#### Features
- **Face Encoding Storage**: Store face encodings efficiently
- **Verification History**: Maintain verification history
- **Performance Metrics**: Track verification performance
- **Data Integrity**: Ensure data integrity

#### Key Functions
```python
def store_face_encoding(user_id, encoding):
    """
    Store face encoding for a user.
    
    Parameters:
    - user_id: User identifier
    - encoding: Face encoding vector
    """

def retrieve_face_encoding(user_id):
    """
    Retrieve face encoding for a user.
    
    Parameters:
    - user_id: User identifier
    
    Returns:
    - Face encoding vector
    """

def verify_face(encoding, threshold=0.6):
    """
    Verify a face against stored encodings.
    
    Parameters:
    - encoding: Face encoding to verify
    - threshold: Verification threshold
    
    Returns:
    - Verification result
    """
```

### Secure Logger (`storage/logger_secure.py`)

#### Purpose
The Secure Logger provides secure logging capabilities with encryption and access control.

#### Features
- **Encrypted Logging**: Encrypt sensitive log data
- **Access Control**: Control access to log files
- **Audit Trail**: Maintain audit trail
- **Data Integrity**: Ensure log data integrity

#### Key Functions
```python
def log_secure_event(event_type, data, user_id=None):
    """
    Log a secure event with encryption.
    
    Parameters:
    - event_type: Type of event
    - data: Event data
    - user_id: User identifier (optional)
    """

def retrieve_secure_logs(user_id, date_range=None):
    """
    Retrieve secure logs for a user.
    
    Parameters:
    - user_id: User identifier
    - date_range: Date range filter
    
    Returns:
    - List of secure log entries
    """
```

## Data Storage Structure

### Face Database (`data/face_db/`)

#### Structure
```
data/face_db/
├── user_001/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── sample_003.jpg
├── user_002/
│   ├── sample_001.jpg
│   └── sample_002.jpg
└── unknown/
    ├── unknown_20240115_143022.jpg
    └── unknown_20240115_143045.jpg
```

#### File Naming Convention
- **User Samples**: `sample_[timestamp].jpg`
- **Unknown Faces**: `unknown_[timestamp].jpg`
- **Timestamps**: Format `YYYYMMDD_HHMMSS`

### Learning Database (`data/learning_db/`)

#### Structure
```
data/learning_db/
├── learning_data.pkl          # Main learning data
├── feature_cache/             # Feature cache
│   ├── user_001_features.pkl
│   └── user_002_features.pkl
└── model_cache/               # Model cache
    ├── face_model.pkl
    └── pose_model.pkl
```

#### Data Format
```python
learning_data = {
    "features": np.array([...]),    # Feature vectors
    "labels": [...],                # Corresponding labels
    "meta": {...}                   # Metadata
}
```

### Model Storage (`data/models/`)

#### Structure
```
data/models/
├── haarcascade_frontalface_default.xml
├── face_recognition_models/
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── pose_models/
│   ├── pose_model.onnx
│   └── pose_model.pkl
└── custom_models/
    ├── mask_detection.pkl
    └── body_analysis.pkl
```

## Backup and Recovery

### Backup Strategy
1. **Incremental Backups**: Daily incremental backups
2. **Full Backups**: Weekly full backups
3. **Offsite Storage**: Offsite backup storage
4. **Version Control**: Version control for models

### Recovery Procedures
1. **Database Recovery**: Restore from SQLite backups
2. **Model Recovery**: Restore AI models
3. **Data Recovery**: Restore face data
4. **System Recovery**: Complete system recovery

### Backup Scripts
```python
def backup_database():
    """
    Backup the main database.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/db_backup_{timestamp}.db"
    shutil.copy2(DB_PATH, backup_path)
    print(f"Database backed up to {backup_path}")

def backup_face_data():
    """
    Backup face database.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/face_data_backup_{timestamp}.zip"
    shutil.make_archive(backup_path, 'zip', FACE_DB_DIR)
    print(f"Face data backed up to {backup_path}")
```

## Performance Optimization

### Database Optimization
1. **Indexing**: Create indexes on frequently queried columns
2. **Query Optimization**: Optimize database queries
3. **Connection Pooling**: Use connection pooling
4. **Caching**: Implement query result caching

### Storage Optimization
1. **Compression**: Compress stored images
2. **Deduplication**: Remove duplicate data
3. **Cleanup**: Regular cleanup of old data
4. **Efficient Formats**: Use efficient storage formats

### Memory Management
1. **Lazy Loading**: Load data only when needed
2. **Memory Limits**: Implement memory limits
3. **Garbage Collection**: Proper garbage collection
4. **Resource Cleanup**: Clean up resources properly

## Security Considerations

### Data Protection
1. **Encryption**: Encrypt sensitive data
2. **Access Control**: Control data access
3. **Audit Logging**: Log all data access
4. **Data Integrity**: Ensure data integrity

### Privacy Compliance
1. **Data Minimization**: Collect only necessary data
2. **Consent Management**: Manage user consent
3. **Data Retention**: Implement retention policies
4. **Right to Deletion**: Support data deletion requests

### Security Measures
1. **Authentication**: Strong authentication
2. **Authorization**: Proper authorization
3. **Encryption**: End-to-end encryption
4. **Monitoring**: Security monitoring

## Troubleshooting

### Common Issues

#### Database Issues
- **Connection Errors**: Check database file permissions
- **Corruption**: Restore from backup
- **Performance**: Optimize queries and indexes
- **Space**: Monitor disk space

#### Storage Issues
- **Permission Errors**: Check file permissions
- **Disk Space**: Monitor disk space usage
- **Corruption**: Verify data integrity
- **Performance**: Optimize storage operations

### Debug Tools
- **Logging**: Comprehensive logging
- **Monitoring**: System monitoring
- **Validation**: Data validation
- **Testing**: Unit and integration tests

## Best Practices

### Data Management
1. **Regular Backups**: Implement regular backups
2. **Data Validation**: Validate all data
3. **Error Handling**: Comprehensive error handling
4. **Monitoring**: Monitor system health

### Security
1. **Access Control**: Implement proper access control
2. **Encryption**: Encrypt sensitive data
3. **Audit Logging**: Maintain audit logs
4. **Regular Updates**: Keep system updated

### Performance
1. **Optimization**: Continuously optimize performance
2. **Monitoring**: Monitor performance metrics
3. **Scaling**: Design for scalability
4. **Resource Management**: Manage resources efficiently

---

This documentation provides comprehensive coverage of the Storage and Database systems, their functionality, and best practices. The system is designed to be secure, efficient, and scalable, providing a solid foundation for data management in the AI Camera system.
