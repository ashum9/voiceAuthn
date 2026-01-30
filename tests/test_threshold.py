"""
VoiceAuth MVP - Threshold Tuning & Testing Script
=================================================
Phase 2: Embedding Comparison & Threshold Implementation

This script helps you:
1. Compare same-speaker pairs (should score > 0.85)
2. Compare different-speaker pairs (should score < 0.70)
3. Test noisy vs clean samples (should score > 0.75)
4. Calculate Equal Error Rate (EER) for optimal threshold

Usage:
    python tests/test_threshold.py                    # Run with synthetic data
    python tests/test_threshold.py --data-dir data/   # Run with real audio files

Updated: January 2026
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    extract_embedding,
    compute_similarity,
    verify_embedding,
    verify_speaker,
    validate_embedding,
    normalize_embedding,
    is_normalized,
    SIMILARITY_THRESHOLD,
    STRICT_THRESHOLD,
    LENIENT_THRESHOLD,
    EMBEDDING_DIM,
    ENABLE_DEBUG_LOGGING,
)


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def generate_synthetic_embedding(speaker_id: int, variation: float = 0.1) -> np.ndarray:
    """
    Generate a synthetic embedding for testing.
    
    Embeddings from the same speaker_id will be similar but not identical.
    Different speaker_ids produce different embeddings.
    
    Args:
        speaker_id: Integer ID to generate consistent speaker "signature"
        variation: Amount of random variation (0-1)
        
    Returns:
        192-dim synthetic embedding
    """
    # Use speaker_id as seed for reproducible base embedding
    np.random.seed(speaker_id * 12345)
    base = np.random.randn(EMBEDDING_DIM)
    base = base / np.linalg.norm(base)  # Normalize
    
    # Add variation (simulates different recordings of same speaker)
    np.random.seed(None)  # Random seed for variation
    noise = np.random.randn(EMBEDDING_DIM) * variation
    embedding = base + noise
    embedding = embedding / np.linalg.norm(embedding)  # Re-normalize
    
    return embedding.astype(np.float32)


def generate_noisy_embedding(base_embedding: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
    """
    Add noise to an embedding to simulate noisy recording conditions.
    
    Args:
        base_embedding: Original embedding
        noise_level: Amount of noise (0-1)
        
    Returns:
        Noisy embedding
    """
    noise = np.random.randn(EMBEDDING_DIM) * noise_level
    noisy = base_embedding + noise
    noisy = noisy / np.linalg.norm(noisy)
    return noisy.astype(np.float32)


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_same_speaker_pairs(n_pairs: int = 20, variation: float = 0.1) -> List[float]:
    """
    Test similarity scores for same-speaker pairs.
    
    Expected: Scores should be > 0.85 for same speaker.
    """
    print(f"\n{'='*60}")
    print("TEST: Same-Speaker Pairs")
    print(f"{'='*60}")
    print(f"Pairs: {n_pairs}, Variation: {variation}")
    print("-" * 60)
    
    scores = []
    for i in range(n_pairs):
        speaker_id = i + 1
        emb1 = generate_synthetic_embedding(speaker_id, variation)
        emb2 = generate_synthetic_embedding(speaker_id, variation)
        
        score = compute_similarity(emb1, emb2)
        scores.append(score)
        
        status = "✓" if score > 0.85 else "⚠" if score > 0.75 else "✗"
        print(f"  Speaker {speaker_id:02d}: {score:.4f} {status}")
    
    # Statistics
    print("-" * 60)
    print(f"Min: {min(scores):.4f}  Max: {max(scores):.4f}  Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    
    above_85 = sum(1 for s in scores if s > 0.85)
    above_75 = sum(1 for s in scores if s > 0.75)
    print(f"Above 0.85: {above_85}/{n_pairs} ({100*above_85/n_pairs:.1f}%)")
    print(f"Above 0.75: {above_75}/{n_pairs} ({100*above_75/n_pairs:.1f}%)")
    
    return scores


def test_different_speaker_pairs(n_pairs: int = 20) -> List[float]:
    """
    Test similarity scores for different-speaker pairs.
    
    Expected: Scores should be < 0.70 for different speakers.
    """
    print(f"\n{'='*60}")
    print("TEST: Different-Speaker Pairs")
    print(f"{'='*60}")
    print(f"Pairs: {n_pairs}")
    print("-" * 60)
    
    scores = []
    for i in range(n_pairs):
        speaker1 = i + 1
        speaker2 = i + 100  # Different speaker
        
        emb1 = generate_synthetic_embedding(speaker1, variation=0.05)
        emb2 = generate_synthetic_embedding(speaker2, variation=0.05)
        
        score = compute_similarity(emb1, emb2)
        scores.append(score)
        
        status = "✓" if score < 0.70 else "⚠" if score < 0.80 else "✗"
        print(f"  Speaker {speaker1:02d} vs {speaker2:03d}: {score:.4f} {status}")
    
    # Statistics
    print("-" * 60)
    print(f"Min: {min(scores):.4f}  Max: {max(scores):.4f}  Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    
    below_70 = sum(1 for s in scores if s < 0.70)
    below_80 = sum(1 for s in scores if s < 0.80)
    print(f"Below 0.70: {below_70}/{n_pairs} ({100*below_70/n_pairs:.1f}%)")
    print(f"Below 0.80: {below_80}/{n_pairs} ({100*below_80/n_pairs:.1f}%)")
    
    return scores


def test_noisy_vs_clean(n_pairs: int = 10, noise_level: float = 0.2) -> List[float]:
    """
    Test similarity between noisy and clean recordings of same speaker.
    
    Expected: Scores should be > 0.75.
    """
    print(f"\n{'='*60}")
    print("TEST: Noisy vs Clean (Same Speaker)")
    print(f"{'='*60}")
    print(f"Pairs: {n_pairs}, Noise Level: {noise_level}")
    print("-" * 60)
    
    scores = []
    for i in range(n_pairs):
        speaker_id = i + 1
        clean = generate_synthetic_embedding(speaker_id, variation=0.05)
        noisy = generate_noisy_embedding(clean, noise_level)
        
        score = compute_similarity(clean, noisy)
        scores.append(score)
        
        status = "✓" if score > 0.75 else "⚠" if score > 0.65 else "✗"
        print(f"  Speaker {speaker_id:02d} (clean vs noisy): {score:.4f} {status}")
    
    # Statistics
    print("-" * 60)
    print(f"Min: {min(scores):.4f}  Max: {max(scores):.4f}  Mean: {np.mean(scores):.4f}")
    
    above_75 = sum(1 for s in scores if s > 0.75)
    print(f"Above 0.75: {above_75}/{n_pairs} ({100*above_75/n_pairs:.1f}%)")
    
    return scores


def calculate_eer(
    same_speaker_scores: List[float],
    diff_speaker_scores: List[float],
    thresholds: Optional[List[float]] = None
) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) and optimal threshold.
    
    EER is where False Accept Rate (FAR) equals False Reject Rate (FRR).
    
    Args:
        same_speaker_scores: Scores from same-speaker comparisons (genuine)
        diff_speaker_scores: Scores from different-speaker comparisons (impostor)
        thresholds: List of thresholds to evaluate (default: 0.50 to 0.95)
        
    Returns:
        Tuple of (EER, optimal_threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.01)
    
    print(f"\n{'='*60}")
    print("EER ANALYSIS")
    print(f"{'='*60}")
    print(f"Genuine pairs: {len(same_speaker_scores)}")
    print(f"Impostor pairs: {len(diff_speaker_scores)}")
    print("-" * 60)
    
    best_eer = 1.0
    best_threshold = 0.80
    
    results = []
    for threshold in thresholds:
        # False Reject Rate: genuine pairs rejected (score < threshold)
        frr = sum(1 for s in same_speaker_scores if s < threshold) / len(same_speaker_scores)
        
        # False Accept Rate: impostor pairs accepted (score >= threshold)
        far = sum(1 for s in diff_speaker_scores if s >= threshold) / len(diff_speaker_scores)
        
        eer_diff = abs(far - frr)
        results.append((threshold, far, frr, eer_diff))
        
        if eer_diff < best_eer:
            best_eer = eer_diff
            best_threshold = threshold
            best_far = far
            best_frr = frr
    
    # Print table
    print(f"{'Threshold':>10} {'FAR':>10} {'FRR':>10} {'|FAR-FRR|':>10}")
    print("-" * 45)
    for threshold, far, frr, diff in results[::5]:  # Every 5th result
        marker = " <-- EER" if abs(threshold - best_threshold) < 0.01 else ""
        print(f"{threshold:>10.2f} {far:>10.2%} {frr:>10.2%} {diff:>10.4f}{marker}")
    
    print("-" * 45)
    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"EER: ~{(best_far + best_frr) / 2:.2%}")
    print(f"FAR at EER: {best_far:.2%}")
    print(f"FRR at EER: {best_frr:.2%}")
    
    return (best_far + best_frr) / 2, best_threshold


def test_verify_embedding_function():
    """Test the verify_embedding function with debug output."""
    print(f"\n{'='*60}")
    print("TEST: verify_embedding() Function")
    print(f"{'='*60}")
    
    # Same speaker - should match
    print("\n[1] Same Speaker Test:")
    emb1 = generate_synthetic_embedding(1, variation=0.08)
    emb2 = generate_synthetic_embedding(1, variation=0.08)
    is_match, score = verify_embedding(emb1, emb2, threshold=0.82, debug=True)
    print(f"Result: {'PASS' if is_match else 'FAIL'}")
    
    # Different speakers - should not match
    print("\n[2] Different Speakers Test:")
    emb3 = generate_synthetic_embedding(2, variation=0.05)
    emb4 = generate_synthetic_embedding(3, variation=0.05)
    is_match, score = verify_embedding(emb3, emb4, threshold=0.82, debug=True)
    print(f"Result: {'PASS (correctly rejected)' if not is_match else 'FAIL (false accept)'}")
    
    # Edge case - identical embeddings
    print("\n[3] Identical Embedding Test:")
    emb5 = generate_synthetic_embedding(5, variation=0.0)
    is_match, score = verify_embedding(emb5, emb5, threshold=0.82, debug=True)
    print(f"Result: {'PASS' if is_match and score > 0.99 else 'FAIL'}")


def test_vector_validation():
    """Test embedding validation and normalization."""
    print(f"\n{'='*60}")
    print("TEST: Vector Validation & Normalization")
    print(f"{'='*60}")
    
    # Test valid embedding
    print("\n[1] Valid embedding:")
    valid = generate_synthetic_embedding(1)
    try:
        validate_embedding(valid)
        print(f"  ✓ Validated successfully")
        print(f"  ✓ Is normalized: {is_normalized(valid)}")
    except ValueError as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test wrong dimension
    print("\n[2] Wrong dimension (100-dim instead of 192):")
    wrong_dim = np.random.randn(100).astype(np.float32)
    try:
        validate_embedding(wrong_dim)
        print(f"  ✗ Should have raised error")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")
    
    # Test NaN values
    print("\n[3] NaN values:")
    nan_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    nan_emb[50] = np.nan
    try:
        validate_embedding(nan_emb)
        print(f"  ✗ Should have raised error")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")
    
    # Test zero vector
    print("\n[4] Zero vector:")
    zero_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        validate_embedding(zero_emb)
        print(f"  ✗ Should have raised error")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")
    
    # Test normalization
    print("\n[5] Normalization:")
    unnormalized = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 5
    print(f"  Original norm: {np.linalg.norm(unnormalized):.4f}")
    normalized = normalize_embedding(unnormalized)
    print(f"  After normalization: {np.linalg.norm(normalized):.4f}")
    print(f"  ✓ Is normalized: {is_normalized(normalized)}")


def test_with_real_audio(data_dir: str):
    """
    Test with real audio files from data directory.
    
    Expected structure:
        data/
            speaker1/
                sample1.wav
                sample2.wav
            speaker2/
                sample1.wav
                sample2.wav
    """
    print(f"\n{'='*60}")
    print("TEST: Real Audio Files")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"  ✗ Directory not found: {data_dir}")
        print("  Create speaker subdirectories with WAV files to test.")
        return [], []
    
    # Find audio files
    speakers = {}
    for root, dirs, files in os.walk(data_dir):
        wav_files = [f for f in files if f.endswith(('.wav', '.mp3', '.flac'))]
        if wav_files:
            speaker = os.path.basename(root)
            speakers[speaker] = [os.path.join(root, f) for f in wav_files]
    
    if not speakers:
        print("  No audio files found. Expected structure:")
        print("    data/speaker1/sample1.wav")
        print("    data/speaker1/sample2.wav")
        print("    data/speaker2/sample1.wav")
        return [], []
    
    print(f"Found {len(speakers)} speakers:")
    for speaker, files in speakers.items():
        print(f"  {speaker}: {len(files)} files")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = {}
    for speaker, files in speakers.items():
        embeddings[speaker] = []
        for f in files[:5]:  # Max 5 per speaker
            try:
                emb = extract_embedding(f)
                embeddings[speaker].append(emb)
                print(f"  ✓ {os.path.basename(f)}")
            except Exception as e:
                print(f"  ✗ {os.path.basename(f)}: {e}")
    
    # Compare same speaker
    same_scores = []
    print("\nSame-speaker comparisons:")
    for speaker, embs in embeddings.items():
        if len(embs) >= 2:
            for i in range(len(embs) - 1):
                score = compute_similarity(embs[i], embs[i+1])
                same_scores.append(score)
                print(f"  {speaker}: {score:.4f}")
    
    # Compare different speakers
    diff_scores = []
    speaker_list = list(embeddings.keys())
    print("\nDifferent-speaker comparisons:")
    for i in range(len(speaker_list)):
        for j in range(i + 1, len(speaker_list)):
            if embeddings[speaker_list[i]] and embeddings[speaker_list[j]]:
                score = compute_similarity(
                    embeddings[speaker_list[i]][0],
                    embeddings[speaker_list[j]][0]
                )
                diff_scores.append(score)
                print(f"  {speaker_list[i]} vs {speaker_list[j]}: {score:.4f}")
    
    return same_scores, diff_scores


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VoiceAuth Threshold Tuning")
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Directory with audio files for testing")
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="Number of pairs to test")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic embedding tests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VoiceAuth MVP - Threshold Tuning & Testing")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Current thresholds:")
    print(f"  SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
    print(f"  STRICT_THRESHOLD:     {STRICT_THRESHOLD}")
    print(f"  LENIENT_THRESHOLD:    {LENIENT_THRESHOLD}")
    print(f"  EMBEDDING_DIM:        {EMBEDDING_DIM}")
    
    # Run tests
    if not args.skip_synthetic:
        # Validation tests
        test_vector_validation()
        
        # verify_embedding function test
        test_verify_embedding_function()
        
        # Same speaker tests
        same_scores = test_same_speaker_pairs(n_pairs=args.n_pairs)
        
        # Different speaker tests
        diff_scores = test_different_speaker_pairs(n_pairs=args.n_pairs)
        
        # Noisy vs clean tests
        noisy_scores = test_noisy_vs_clean(n_pairs=10)
        
        # EER calculation
        eer, optimal_threshold = calculate_eer(same_scores, diff_scores)
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"1. Set SIMILARITY_THRESHOLD to {optimal_threshold:.2f}")
        print(f"2. For high security, use STRICT_THRESHOLD = {min(0.90, optimal_threshold + 0.05):.2f}")
        print(f"3. For demos/noisy environments, use LENIENT_THRESHOLD = {max(0.65, optimal_threshold - 0.10):.2f}")
    
    # Real audio tests
    if os.path.exists(args.data_dir):
        real_same, real_diff = test_with_real_audio(args.data_dir)
        if real_same and real_diff:
            print("\nReal Audio EER:")
            calculate_eer(real_same, real_diff)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
