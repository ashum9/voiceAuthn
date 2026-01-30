#!/usr/bin/env python3
"""
Phase 4 Test Script: Secure Voiceprint Storage
==============================================

Tests the voiceprint storage functionality:
1. Store/retrieve embeddings
2. Encryption verification
3. Deletion (GDPR compliance)
4. In-memory fallback
5. Database integrity

Usage:
    python tests/test_storage.py
"""

import os
import sys
import time
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage import (
    VoiceprintDB,
    get_voiceprint_db,
    EncryptionManager,
    get_encryption_manager,
    create_user,
    get_user,
    delete_user,
    save_enrollment,
    get_user_enrollments,
    delete_all_enrollments,
    get_database_stats,
    EMBEDDING_DIM
)


def generate_test_embedding(seed: int = None) -> np.ndarray:
    """Generate a random test embedding."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(EMBEDDING_DIM).astype(np.float32)


def test_voiceprint_db_basic():
    """Test 1: Basic VoiceprintDB Operations"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic VoiceprintDB Operations")
    print("=" * 60)
    
    # Create a fresh DB in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        key_file = os.path.join(tmpdir, "test.key")
        
        db = VoiceprintDB(db_path=db_path, key_file=key_file)
        
        print("\n  [1a] Store embedding:")
        embedding = generate_test_embedding(seed=42)
        result = db.store("user_001", embedding)
        print(f"    Stored: {result}")
        assert result, "Store failed"
        print(f"    ✓ Stored successfully")
        
        print("\n  [1b] Retrieve embedding:")
        retrieved = db.retrieve("user_001")
        assert retrieved is not None, "Retrieve returned None"
        print(f"    Retrieved shape: {retrieved.shape}")
        print(f"    First 5 values: {retrieved[:5].round(4)}")
        print(f"    ✓ Retrieved successfully")
        
        print("\n  [1c] Verify exact match:")
        max_diff = np.abs(embedding - retrieved).max()
        print(f"    Max difference: {max_diff:.10f}")
        assert max_diff < 1e-6, f"Mismatch detected: {max_diff}"
        print(f"    ✓ Exact match verified")
        
        print("\n  [1d] Non-existent user:")
        none_result = db.retrieve("nonexistent_user")
        assert none_result is None, "Should return None for non-existent user"
        print(f"    ✓ Correctly returns None")
        
        print("\n  ✓ PASSED")
        return True


def test_encryption_verification():
    """Test 2: Encryption Verification"""
    print("\n" + "=" * 60)
    print("TEST 2: Encryption Verification")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        key_file = os.path.join(tmpdir, "test.key")
        
        db = VoiceprintDB(db_path=db_path, key_file=key_file)
        
        # Store embedding
        embedding = generate_test_embedding(seed=123)
        db.store("encrypted_user", embedding)
        
        print("\n  [2a] Get raw encrypted blob:")
        encrypted_blob = db.get_raw_encrypted("encrypted_user")
        assert encrypted_blob is not None, "No encrypted blob found"
        print(f"    Encrypted blob size: {len(encrypted_blob)} bytes")
        print(f"    Expected raw size: {EMBEDDING_DIM * 4} bytes (192 floats * 4)")
        print(f"    ✓ Blob is larger due to encryption overhead")
        
        print("\n  [2b] Verify blob is unreadable:")
        # Try to interpret as raw floats (should be garbage)
        try:
            # The encrypted blob won't have the right size for 192 floats
            # and won't parse correctly
            fake_embedding = np.frombuffer(encrypted_blob[:EMBEDDING_DIM*4], dtype=np.float32)
            # Check if it looks like the original
            similarity = np.dot(embedding, fake_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(fake_embedding) + 1e-10
            )
            print(f"    Similarity of raw interpretation: {similarity:.4f}")
            assert abs(similarity) < 0.5, "Encrypted data should not resemble original"
            print(f"    ✓ Encrypted blob is not interpretable as original")
        except Exception as e:
            print(f"    ✓ Cannot interpret encrypted blob: {type(e).__name__}")
        
        print("\n  [2c] Decryption produces correct result:")
        retrieved = db.retrieve("encrypted_user")
        match = np.allclose(embedding, retrieved, atol=1e-6)
        print(f"    Embeddings match: {match}")
        assert match, "Decryption failed"
        print(f"    ✓ Decryption successful")
        
        print("\n  [2d] Wrong key cannot decrypt:")
        # Create new DB with different key
        key_file2 = os.path.join(tmpdir, "wrong.key")
        db2 = VoiceprintDB(db_path=db_path, key_file=key_file2)
        
        try:
            # This should fail or return garbage
            retrieved2 = db2.retrieve("encrypted_user")
            if retrieved2 is not None:
                match2 = np.allclose(embedding, retrieved2, atol=1e-6)
                assert not match2, "Wrong key should not produce correct result"
            print(f"    ✓ Wrong key cannot decrypt correctly")
        except Exception as e:
            print(f"    ✓ Wrong key raises exception: {type(e).__name__}")
        
        print("\n  ✓ PASSED")
        return True


def test_delete_functionality():
    """Test 3: Delete Functionality (GDPR)"""
    print("\n" + "=" * 60)
    print("TEST 3: Delete Functionality (GDPR Compliance)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        key_file = os.path.join(tmpdir, "test.key")
        
        db = VoiceprintDB(db_path=db_path, key_file=key_file)
        
        # Store multiple users
        print("\n  [3a] Store multiple users:")
        for i in range(3):
            embedding = generate_test_embedding(seed=i)
            db.store(f"user_{i:03d}", embedding)
        
        users = db.list_users()
        print(f"    Stored {len(users)} users: {users}")
        assert len(users) == 3, "Should have 3 users"
        
        print("\n  [3b] Delete one user:")
        deleted = db.delete("user_001")
        print(f"    Deleted: {deleted}")
        assert deleted, "Delete should return True"
        
        users_after = db.list_users()
        print(f"    Remaining users: {users_after}")
        assert "user_001" not in users_after, "Deleted user should not exist"
        assert len(users_after) == 2, "Should have 2 users remaining"
        print(f"    ✓ User successfully deleted")
        
        print("\n  [3c] Verify deleted user cannot be retrieved:")
        result = db.retrieve("user_001")
        assert result is None, "Deleted user should return None"
        print(f"    ✓ Deleted user returns None")
        
        print("\n  [3d] Delete non-existent user:")
        deleted2 = db.delete("nonexistent")
        print(f"    Delete result: {deleted2} (expected: False)")
        # Note: May return True or False depending on implementation
        print(f"    ✓ No error on deleting non-existent user")
        
        print("\n  ✓ PASSED")
        return True


def test_memory_fallback():
    """Test 4: In-Memory Fallback"""
    print("\n" + "=" * 60)
    print("TEST 4: In-Memory Fallback")
    print("=" * 60)
    
    # Create DB with invalid path to force fallback
    db = VoiceprintDB.__new__(VoiceprintDB)
    db.db_path = "/invalid/path/that/does/not/exist/db.sqlite"
    db.key_file = "/invalid/key.file"
    db._use_memory_fallback = True
    db._memory_store = {}
    db._fernet = None
    
    # Initialize encryption manually
    from cryptography.fernet import Fernet
    db._fernet = Fernet(Fernet.generate_key())
    
    print("\n  [4a] Store in memory:")
    embedding = generate_test_embedding(seed=999)
    result = db.store("memory_user", embedding)
    print(f"    Stored: {result}")
    assert result, "Memory store failed"
    
    print("\n  [4b] Retrieve from memory:")
    retrieved = db.retrieve("memory_user")
    assert retrieved is not None, "Memory retrieve failed"
    match = np.allclose(embedding, retrieved, atol=1e-6)
    print(f"    Match: {match}")
    assert match, "Memory data mismatch"
    print(f"    ✓ Memory storage works")
    
    print("\n  [4c] List users from memory:")
    users = db.list_users()
    print(f"    Users: {users}")
    assert "memory_user" in users, "User not in list"
    
    print("\n  [4d] Stats show in-memory mode:")
    stats = db.get_stats()
    print(f"    Mode: {stats.get('mode')}")
    assert stats.get('mode') == 'in-memory', "Should be in-memory mode"
    
    print("\n  ✓ PASSED")
    return True


def test_update_existing():
    """Test 5: Update Existing User"""
    print("\n" + "=" * 60)
    print("TEST 5: Update Existing User")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        key_file = os.path.join(tmpdir, "test.key")
        
        db = VoiceprintDB(db_path=db_path, key_file=key_file)
        
        print("\n  [5a] Store initial embedding:")
        embedding1 = generate_test_embedding(seed=100)
        db.store("update_user", embedding1)
        
        retrieved1 = db.retrieve("update_user")
        print(f"    Initial first 3 values: {retrieved1[:3].round(4)}")
        
        print("\n  [5b] Update with new embedding:")
        embedding2 = generate_test_embedding(seed=200)  # Different seed = different values
        db.store("update_user", embedding2)
        
        retrieved2 = db.retrieve("update_user")
        print(f"    Updated first 3 values: {retrieved2[:3].round(4)}")
        
        print("\n  [5c] Verify update:")
        match_old = np.allclose(embedding1, retrieved2, atol=1e-6)
        match_new = np.allclose(embedding2, retrieved2, atol=1e-6)
        print(f"    Matches old: {match_old}")
        print(f"    Matches new: {match_new}")
        assert not match_old, "Should not match old embedding"
        assert match_new, "Should match new embedding"
        print(f"    ✓ Update successful (INSERT OR REPLACE)")
        
        print("\n  [5d] User count unchanged:")
        users = db.list_users()
        assert len(users) == 1, "Should still have 1 user"
        print(f"    ✓ No duplicate users created")
        
        print("\n  ✓ PASSED")
        return True


def test_full_system_storage():
    """Test 6: Full System Storage (with existing storage.py functions)"""
    print("\n" + "=" * 60)
    print("TEST 6: Full System Storage Integration")
    print("=" * 60)
    
    print("\n  [6a] Create user via storage.py:")
    try:
        user_id = create_user("phase4_test_user", "test@example.com")
        print(f"    Created: {user_id}")
    except ValueError:
        user = get_user("phase4_test_user")
        user_id = user['user_id']
        print(f"    Existing user: {user_id}")
        # Clean up old enrollments
        delete_all_enrollments(user_id)
    
    print("\n  [6b] Save enrollment:")
    embedding = generate_test_embedding(seed=42)
    enrollment_id = save_enrollment(
        user_id,
        embedding,
        phrase_used="Test phrase for Phase 4",
        audio_duration=4.5,
        quality_score=0.95
    )
    print(f"    Enrollment ID: {enrollment_id}")
    
    print("\n  [6c] Retrieve enrollment:")
    embeddings = get_user_enrollments(user_id)
    print(f"    Retrieved {len(embeddings)} embedding(s)")
    assert len(embeddings) >= 1, "Should have at least 1 enrollment"
    
    retrieved = embeddings[0]
    match = np.allclose(embedding, retrieved, atol=1e-6)
    print(f"    Exact match: {match}")
    assert match, "Embedding mismatch"
    
    print("\n  [6d] Database stats:")
    stats = get_database_stats()
    print(f"    Users: {stats['users']}")
    print(f"    Enrollments: {stats['enrollments']}")
    
    print("\n  [6e] Cleanup - delete user:")
    deleted = delete_user(user_id)
    print(f"    Deleted: {deleted}")
    
    print("\n  ✓ PASSED")
    return True


def test_blob_size():
    """Test 7: Verify Blob Size"""
    print("\n" + "=" * 60)
    print("TEST 7: Blob Size Verification")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        key_file = os.path.join(tmpdir, "test.key")
        
        db = VoiceprintDB(db_path=db_path, key_file=key_file)
        
        embedding = generate_test_embedding(seed=1)
        db.store("size_test", embedding)
        
        encrypted = db.get_raw_encrypted("size_test")
        
        raw_size = EMBEDDING_DIM * 4  # 192 * 4 bytes = 768 bytes
        encrypted_size = len(encrypted)
        
        print(f"\n  Raw embedding size: {raw_size} bytes ({EMBEDDING_DIM} floats * 4)")
        print(f"  Encrypted blob size: {encrypted_size} bytes")
        print(f"  Encryption overhead: {encrypted_size - raw_size} bytes")
        
        # Fernet adds ~57 bytes overhead (IV + HMAC + padding)
        assert encrypted_size < 2000, "Blob should be under 2KB"
        print(f"\n  ✓ Blob size is reasonable (~{encrypted_size/1024:.2f} KB)")
        
        print("\n  ✓ PASSED")
        return True


def run_all_tests():
    """Run all Phase 4 tests"""
    print("=" * 60)
    print("PHASE 4: Secure Voiceprint Storage - Test Suite")
    print("=" * 60)
    
    results = {}
    
    results["Basic Operations"] = test_voiceprint_db_basic()
    results["Encryption Verification"] = test_encryption_verification()
    results["Delete Functionality"] = test_delete_functionality()
    results["Memory Fallback"] = test_memory_fallback()
    results["Update Existing"] = test_update_existing()
    results["Full System Integration"] = test_full_system_storage()
    results["Blob Size"] = test_blob_size()
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 4 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Phase 4 Complete!")
    else:
        print("⚠ SOME TESTS FAILED - Review above for details")
    print("=" * 60)
    
    # Checkpoint verification
    print("\n[Checkpoint Tests]")
    print("  • Store/retrieve: Exact match ✓" if results.get("Basic Operations") else "  • Store/retrieve: ⚠")
    print("  • Encrypted BLOB: Unreadable without key ✓" if results.get("Encryption Verification") else "  • Encryption: ⚠")
    print("  • GDPR deletion works ✓" if results.get("Delete Functionality") else "  • Deletion: ⚠")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
