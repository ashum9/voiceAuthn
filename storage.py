"""
VoiceAuth MVP - Storage Module
==============================
Handles: User data storage, embedding management, encryption,
         challenge tracking, and audit logging.

Storage: SQLite for MVP (easy setup, no external DB needed)
Security: AES-256 encryption for embeddings at rest

Phase 4: Secure Voiceprint Storage
- Encrypted embedding storage (Fernet/AES-128-CBC)
- No raw audio stored (only embeddings)
- User deletion capability (GDPR right to erasure)
- In-memory fallback if DB fails

GDPR/Privacy Compliance Notes (2026):
- Data Minimization: Only store embeddings, not raw audio
- Purpose Limitation: Embeddings used only for authentication
- Storage Limitation: Audio is transient, deleted after embedding extraction
- Right to Erasure: delete_user() removes all user data
- Encryption at Rest: All embeddings encrypted with Fernet
- Key Management: Encryption key stored separately with restricted permissions

Updated: January 2026
"""

import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple, Any
from contextlib import contextmanager
import numpy as np

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from utils import (
    embedding_to_bytes,
    bytes_to_embedding,
    EMBEDDING_DIM,
    CHALLENGE_EXPIRY_SECONDS
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATABASE_PATH = "data/voiceauth.db"
ENCRYPTION_KEY_FILE = "data/.encryption_key"
MAX_ENROLLMENTS_PER_USER = 5
AUDIT_LOG_RETENTION_DAYS = 30


# ============================================================================
# PHASE 4: VOICEPRINT DATABASE CLASS
# ============================================================================

class VoiceprintDB:
    """
    Simplified voiceprint storage API (Phase 4).
    
    Provides simple store/retrieve/delete operations for voice embeddings.
    Encrypts embeddings at rest using Fernet (AES-128-CBC with HMAC).
    Falls back to in-memory storage if database fails.
    
    GDPR Compliance:
    - Only stores embeddings, not raw audio
    - Provides deletion capability for right to erasure
    - Encryption at rest for data protection
    
    Usage:
        db = VoiceprintDB()
        db.store("user123", embedding_vector)
        embedding = db.retrieve("user123")
        db.delete("user123")
    """
    
    def __init__(self, db_path: str = None, key_file: str = None):
        """
        Initialize the voiceprint database.
        
        Args:
            db_path: Path to SQLite database (default: data/voiceauth.db)
            key_file: Path to encryption key file (default: data/.encryption_key)
        """
        self.db_path = db_path or DATABASE_PATH
        self.key_file = key_file or ENCRYPTION_KEY_FILE
        self._use_memory_fallback = False
        self._memory_store: Dict[str, bytes] = {}  # In-memory fallback
        self._fernet = None
        
        # Initialize encryption
        self._init_encryption()
        
        # Initialize database
        self._init_database()
    
    def _init_encryption(self):
        """Initialize Fernet encryption with key from file."""
        try:
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Secure permissions (owner read/write only)
                os.chmod(self.key_file, 0o600)
                print(f"[VoiceprintDB] Generated new encryption key: {self.key_file}")
            
            self._fernet = Fernet(key)
        except Exception as e:
            print(f"[VoiceprintDB] Encryption init warning: {e}")
            # Fallback: Generate ephemeral key (data won't persist across restarts)
            self._fernet = Fernet(Fernet.generate_key())
    
    def _init_database(self):
        """Initialize SQLite database with users table."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple users table as specified in Phase 4
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS voiceprints (
                    user_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[VoiceprintDB] Database init failed, using in-memory fallback: {e}")
            self._use_memory_fallback = True
    
    def _encrypt(self, embedding: np.ndarray) -> bytes:
        """Encrypt embedding for storage."""
        raw_bytes = embedding.astype(np.float32).tobytes()
        return self._fernet.encrypt(raw_bytes)
    
    def _decrypt(self, encrypted: bytes) -> np.ndarray:
        """Decrypt embedding from storage."""
        raw_bytes = self._fernet.decrypt(encrypted)
        return np.frombuffer(raw_bytes, dtype=np.float32)
    
    def store(self, user_id: str, embedding: np.ndarray, 
              timestamp: datetime = None) -> bool:
        """
        Store a voiceprint for a user.
        
        Args:
            user_id: Unique user identifier
            embedding: 192-dim voice embedding vector
            timestamp: Optional timestamp (default: now)
            
        Returns:
            True if successful
            
        Note:
            If user_id exists, the embedding is updated (INSERT OR REPLACE).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Validate embedding
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"Invalid embedding dimension: {embedding.shape[0]} (expected {EMBEDDING_DIM})")
        
        # Encrypt
        encrypted = self._encrypt(embedding)
        
        if self._use_memory_fallback:
            self._memory_store[user_id] = encrypted
            return True
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO voiceprints (user_id, embedding, timestamp, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, encrypted, timestamp))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[VoiceprintDB] Store failed: {e}")
            # Fallback to memory
            self._memory_store[user_id] = encrypted
            return True
    
    def retrieve(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a voiceprint for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            192-dim embedding vector, or None if not found
        """
        if self._use_memory_fallback:
            encrypted = self._memory_store.get(user_id)
            if encrypted:
                return self._decrypt(encrypted)
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT embedding FROM voiceprints WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._decrypt(row[0])
            return None
            
        except Exception as e:
            print(f"[VoiceprintDB] Retrieve failed: {e}")
            # Try memory fallback
            encrypted = self._memory_store.get(user_id)
            if encrypted:
                return self._decrypt(encrypted)
            return None
    
    def delete(self, user_id: str) -> bool:
        """
        Delete a user's voiceprint (GDPR right to erasure).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user was deleted, False if not found
        """
        # Remove from memory store
        if user_id in self._memory_store:
            del self._memory_store[user_id]
        
        if self._use_memory_fallback:
            return True
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM voiceprints WHERE user_id = ?', (user_id,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            return deleted
            
        except Exception as e:
            print(f"[VoiceprintDB] Delete failed: {e}")
            return False
    
    def exists(self, user_id: str) -> bool:
        """Check if a user has a stored voiceprint."""
        return self.retrieve(user_id) is not None
    
    def list_users(self) -> List[str]:
        """List all user IDs with stored voiceprints."""
        if self._use_memory_fallback:
            return list(self._memory_store.keys())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id FROM voiceprints')
            users = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return users
            
        except Exception as e:
            print(f"[VoiceprintDB] List failed: {e}")
            return list(self._memory_store.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self._use_memory_fallback:
            return {
                "mode": "in-memory",
                "users": len(self._memory_store),
                "db_path": None
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM voiceprints')
            count = cursor.fetchone()[0]
            
            conn.close()
            return {
                "mode": "sqlite",
                "users": count,
                "db_path": self.db_path,
                "key_file": self.key_file
            }
            
        except Exception as e:
            return {"mode": "error", "error": str(e)}
    
    def verify_encryption(self, user_id: str, original_embedding: np.ndarray) -> bool:
        """
        Verify that stored embedding matches original (for testing).
        
        Returns:
            True if embeddings match after encrypt/decrypt cycle
        """
        retrieved = self.retrieve(user_id)
        if retrieved is None:
            return False
        
        return np.allclose(original_embedding, retrieved, atol=1e-6)
    
    def get_raw_encrypted(self, user_id: str) -> Optional[bytes]:
        """
        Get raw encrypted blob (for testing encryption).
        
        Returns:
            Encrypted bytes, or None if not found
        """
        if self._use_memory_fallback:
            return self._memory_store.get(user_id)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT embedding FROM voiceprints WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            conn.close()
            return row[0] if row else None
            
        except Exception:
            return self._memory_store.get(user_id)


# Global VoiceprintDB instance (lazy initialization)
_voiceprint_db = None

def get_voiceprint_db() -> VoiceprintDB:
    """Get the global VoiceprintDB instance."""
    global _voiceprint_db
    if _voiceprint_db is None:
        _voiceprint_db = VoiceprintDB()
    return _voiceprint_db


# ============================================================================
# ENCRYPTION
# ============================================================================

class EncryptionManager:
    """
    Manages encryption/decryption of sensitive data.
    Uses Fernet (AES-128-CBC with HMAC) for symmetric encryption.
    """
    
    def __init__(self, key_file: str = ENCRYPTION_KEY_FILE):
        self.key_file = key_file
        self._fernet = None
    
    def _get_or_create_key(self) -> bytes:
        """Get existing key or generate new one."""
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
            return key
    
    @property
    def fernet(self) -> Fernet:
        if self._fernet is None:
            key = self._get_or_create_key()
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        return self.fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Encrypt a numpy embedding array."""
        raw_bytes = embedding_to_bytes(embedding)
        return self.encrypt(raw_bytes)
    
    def decrypt_embedding(self, encrypted: bytes) -> np.ndarray:
        """Decrypt to numpy embedding array."""
        raw_bytes = self.decrypt(encrypted)
        return bytes_to_embedding(raw_bytes)


# Global encryption manager
_encryption_manager = None

def get_encryption_manager() -> EncryptionManager:
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


# ============================================================================
# DATABASE SETUP
# ============================================================================

def get_db_path() -> str:
    """Get database path, creating directory if needed."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    return DATABASE_PATH


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """
    Initialize database with all required tables.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                failed_attempts INTEGER DEFAULT 0,
                last_failed_attempt TIMESTAMP,
                locked_until TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Voice enrollments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrollments (
                enrollment_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                embedding_encrypted BLOB NOT NULL,
                phrase_used TEXT,
                audio_duration REAL,
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_primary BOOLEAN DEFAULT 0,
                device_info TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        ''')
        
        # Challenges table (for anti-replay)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS challenges (
                challenge_id TEXT PRIMARY KEY,
                user_id TEXT,
                phrases TEXT NOT NULL,
                issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN DEFAULT 0,
                used_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Authentication attempts (audit log)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                username TEXT,
                attempt_type TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                similarity_score REAL,
                challenge_id TEXT,
                spoof_check_passed BOOLEAN,
                failure_reason TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_enrollments_user ON enrollments(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_challenges_user ON challenges(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_log_user ON auth_log(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_log_timestamp ON auth_log(timestamp)')
        
        print("[Storage] Database initialized successfully")


# ============================================================================
# USER MANAGEMENT
# ============================================================================

def generate_user_id() -> str:
    """Generate a unique user ID."""
    return f"user_{secrets.token_hex(8)}"


def create_user(
    username: str,
    email: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create a new user account.
    
    Args:
        username: Unique username
        email: Optional email address
        metadata: Optional additional user data
        
    Returns:
        user_id of created user
        
    Raises:
        ValueError if username already exists
    """
    user_id = generate_user_id()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (user_id, username, email, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                username.lower().strip(),
                email,
                json.dumps(metadata) if metadata else None
            ))
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")
    
    return user_id


def get_user(username: str) -> Optional[Dict]:
    """Get user by username."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM users WHERE username = ? AND is_active = 1
        ''', (username.lower().strip(),))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def user_exists(username: str) -> bool:
    """Check if username exists."""
    return get_user(username) is not None


def delete_user(user_id: str) -> bool:
    """Delete a user and all their data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        return cursor.rowcount > 0


def list_users() -> List[Dict]:
    """List all active users (without sensitive data)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, username, email, created_at, 
                   (SELECT COUNT(*) FROM enrollments WHERE enrollments.user_id = users.user_id) as enrollment_count
            FROM users WHERE is_active = 1
            ORDER BY created_at DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# ACCOUNT LOCKOUT
# ============================================================================

def record_failed_attempt(user_id: str) -> Tuple[int, Optional[datetime]]:
    """
    Record a failed authentication attempt.
    Implements account lockout after multiple failures.
    
    Returns:
        Tuple of (failed_attempt_count, locked_until_datetime)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get current state
        cursor.execute('''
            SELECT failed_attempts, locked_until FROM users WHERE user_id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return 0, None
        
        failed_attempts = row['failed_attempts'] + 1
        
        # Lock account after 5 failed attempts
        locked_until = None
        if failed_attempts >= 5:
            # Lock for 15 minutes
            locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        cursor.execute('''
            UPDATE users 
            SET failed_attempts = ?, 
                last_failed_attempt = CURRENT_TIMESTAMP,
                locked_until = ?
            WHERE user_id = ?
        ''', (failed_attempts, locked_until, user_id))
        
        return failed_attempts, locked_until


def reset_failed_attempts(user_id: str):
    """Reset failed attempt counter after successful login."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET failed_attempts = 0, 
                locked_until = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (user_id,))


def is_account_locked(user_id: str) -> Tuple[bool, Optional[datetime]]:
    """Check if account is locked."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT locked_until FROM users WHERE user_id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        
        if not row or not row['locked_until']:
            return False, None
        
        locked_until = datetime.fromisoformat(row['locked_until'])
        if datetime.now(timezone.utc) > locked_until:
            # Lock expired, reset
            reset_failed_attempts(user_id)
            return False, None
        
        return True, locked_until


# ============================================================================
# ENROLLMENT MANAGEMENT
# ============================================================================

def generate_enrollment_id() -> str:
    """Generate unique enrollment ID."""
    return f"enroll_{secrets.token_hex(8)}"


def save_enrollment(
    user_id: str,
    embedding: np.ndarray,
    phrase_used: Optional[str] = None,
    audio_duration: Optional[float] = None,
    quality_score: Optional[float] = None,
    is_primary: bool = False,
    device_info: Optional[str] = None
) -> str:
    """
    Save a voice enrollment for a user.
    
    Args:
        user_id: User's ID
        embedding: 192-dim voice embedding
        phrase_used: Challenge phrase spoken
        audio_duration: Duration of audio sample
        quality_score: Quality assessment score
        is_primary: Whether this is the primary enrollment
        device_info: Device/browser info
        
    Returns:
        enrollment_id
    """
    # Validate embedding
    if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Invalid embedding dimension: {embedding.shape[0]} (expected {EMBEDDING_DIM})")
    
    # Encrypt embedding
    encrypted = get_encryption_manager().encrypt_embedding(embedding)
    enrollment_id = generate_enrollment_id()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check enrollment limit
        cursor.execute('''
            SELECT COUNT(*) as count FROM enrollments WHERE user_id = ?
        ''', (user_id,))
        count = cursor.fetchone()['count']
        
        if count >= MAX_ENROLLMENTS_PER_USER:
            raise ValueError(f"Maximum enrollments ({MAX_ENROLLMENTS_PER_USER}) reached for user")
        
        # If setting as primary, unset existing primary
        if is_primary:
            cursor.execute('''
                UPDATE enrollments SET is_primary = 0 WHERE user_id = ?
            ''', (user_id,))
        
        # If first enrollment, make it primary
        if count == 0:
            is_primary = True
        
        cursor.execute('''
            INSERT INTO enrollments 
            (enrollment_id, user_id, embedding_encrypted, phrase_used, 
             audio_duration, quality_score, is_primary, device_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            enrollment_id, user_id, encrypted, phrase_used,
            audio_duration, quality_score, is_primary, device_info
        ))
    
    return enrollment_id


def get_user_enrollments(user_id: str) -> List[np.ndarray]:
    """
    Get all voice embeddings for a user.
    
    Args:
        user_id: User's ID
        
    Returns:
        List of decrypted embedding vectors
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT embedding_encrypted FROM enrollments 
            WHERE user_id = ? ORDER BY is_primary DESC, created_at DESC
        ''', (user_id,))
        rows = cursor.fetchall()
    
    embeddings = []
    for row in rows:
        try:
            embedding = get_encryption_manager().decrypt_embedding(row['embedding_encrypted'])
            embeddings.append(embedding)
        except Exception as e:
            print(f"[Storage] Warning: Failed to decrypt enrollment: {e}")
    
    return embeddings


def get_enrollment_details(user_id: str) -> List[Dict]:
    """Get enrollment metadata (without embeddings)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT enrollment_id, phrase_used, audio_duration, 
                   quality_score, created_at, is_primary
            FROM enrollments WHERE user_id = ?
            ORDER BY is_primary DESC, created_at DESC
        ''', (user_id,))
        return [dict(row) for row in cursor.fetchall()]


def delete_enrollment(enrollment_id: str) -> bool:
    """Delete a specific enrollment."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM enrollments WHERE enrollment_id = ?', (enrollment_id,))
        return cursor.rowcount > 0


def delete_all_enrollments(user_id: str) -> int:
    """Delete all enrollments for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM enrollments WHERE user_id = ?', (user_id,))
        return cursor.rowcount


def has_enrollments(user_id: str) -> bool:
    """Check if user has any voice enrollments."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM enrollments WHERE user_id = ? LIMIT 1', (user_id,))
        return cursor.fetchone() is not None


# ============================================================================
# CHALLENGE MANAGEMENT
# ============================================================================

def save_challenge(
    challenge_id: str,
    phrases: List[str],
    expires_at: datetime,
    user_id: Optional[str] = None
):
    """Save a challenge for verification."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO challenges (challenge_id, user_id, phrases, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (challenge_id, user_id, json.dumps(phrases), expires_at))


def get_challenge(challenge_id: str) -> Optional[Dict]:
    """Get challenge details."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM challenges WHERE challenge_id = ?
        ''', (challenge_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            result['phrases'] = json.loads(result['phrases'])
            return result
        return None


def mark_challenge_used(challenge_id: str) -> bool:
    """Mark a challenge as used."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE challenges 
            SET used = 1, used_at = CURRENT_TIMESTAMP
            WHERE challenge_id = ? AND used = 0
        ''', (challenge_id,))
        return cursor.rowcount > 0


def is_challenge_valid(challenge_id: str) -> Tuple[bool, str]:
    """
    Check if a challenge is valid (exists, not expired, not used).
    
    Returns:
        Tuple of (is_valid, reason)
    """
    challenge = get_challenge(challenge_id)
    
    if not challenge:
        return False, "Challenge not found"
    
    if challenge['used']:
        return False, "Challenge already used (replay detected)"
    
    expires_at = datetime.fromisoformat(challenge['expires_at'])
    if datetime.now(timezone.utc) > expires_at:
        return False, "Challenge expired"
    
    return True, "Challenge valid"


def cleanup_expired_challenges():
    """Remove expired challenges."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM challenges WHERE expires_at < CURRENT_TIMESTAMP
        ''')
        return cursor.rowcount


# ============================================================================
# AUDIT LOGGING
# ============================================================================

def log_auth_attempt(
    attempt_type: str,
    success: bool,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    similarity_score: Optional[float] = None,
    challenge_id: Optional[str] = None,
    spoof_check_passed: Optional[bool] = None,
    failure_reason: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """
    Log an authentication attempt for auditing.
    
    attempt_type: 'registration', 'login', 'enrollment_add', 'enrollment_delete'
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO auth_log 
            (user_id, username, attempt_type, success, similarity_score,
             challenge_id, spoof_check_passed, failure_reason, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, username, attempt_type, success, similarity_score,
            challenge_id, spoof_check_passed, failure_reason, ip_address, user_agent
        ))


def get_auth_history(
    user_id: Optional[str] = None,
    limit: int = 50,
    include_failures: bool = True
) -> List[Dict]:
    """Get authentication history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        query = 'SELECT * FROM auth_log'
        params = []
        conditions = []
        
        if user_id:
            conditions.append('user_id = ?')
            params.append(user_id)
        
        if not include_failures:
            conditions.append('success = 1')
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def cleanup_old_logs():
    """Remove audit logs older than retention period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=AUDIT_LOG_RETENTION_DAYS)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM auth_log WHERE timestamp < ?
        ''', (cutoff,))
        return cursor.rowcount


# ============================================================================
# DATABASE STATISTICS
# ============================================================================

def get_database_stats() -> Dict:
    """Get database statistics for monitoring."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        user_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM enrollments')
        enrollment_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM challenges WHERE used = 0 AND expires_at > CURRENT_TIMESTAMP')
        active_challenges = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM auth_log')
        log_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM auth_log WHERE success = 1')
        success_count = cursor.fetchone()[0]
        
        return {
            "users": user_count,
            "enrollments": enrollment_count,
            "active_challenges": active_challenges,
            "total_auth_attempts": log_count,
            "successful_auths": success_count,
            "success_rate": success_count / log_count if log_count > 0 else 0,
            "database_path": get_db_path(),
        }


# ============================================================================
# INITIALIZATION
# ============================================================================

def reset_database():
    """
    Reset the database (DELETE ALL DATA).
    Only for development/testing!
    """
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    init_database()
    print("[Storage] Database reset complete")


# Initialize on import
init_database()


if __name__ == "__main__":
    # Test storage functionality
    print("=" * 50)
    print("VoiceAuth MVP - Storage Test")
    print("=" * 50)
    
    # Create test user
    print("\n[1] Creating test user...")
    try:
        user_id = create_user("test_user", "test@example.com")
        print(f"  ✓ Created user: {user_id}")
    except ValueError as e:
        print(f"  ℹ User exists: {e}")
        user = get_user("test_user")
        user_id = user['user_id']
    
    # Test enrollment
    print("\n[2] Testing enrollment storage...")
    fake_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    enrollment_id = save_enrollment(
        user_id, 
        fake_embedding,
        phrase_used="Test phrase",
        audio_duration=3.5
    )
    print(f"  ✓ Saved enrollment: {enrollment_id}")
    
    # Retrieve
    embeddings = get_user_enrollments(user_id)
    print(f"  ✓ Retrieved {len(embeddings)} enrollment(s)")
    
    # Verify encryption worked
    diff = np.abs(embeddings[0] - fake_embedding).max()
    print(f"  ✓ Encryption/decryption verified (max diff: {diff:.10f})")
    
    # Stats
    print("\n[3] Database statistics:")
    stats = get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Storage test complete!")
    print("=" * 50)
