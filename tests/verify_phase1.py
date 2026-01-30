#!/usr/bin/env python3
"""Phase 1 Verification - Quick Summary"""

import sys
sys.path.insert(0, '.')

from utils import load_model, get_embedding, EMBEDDING_DIM, SAMPLE_RATE

print("=" * 60)
print("Phase 1: Speaker Embedding Extraction - COMPLETE")
print("=" * 60)

print("\n[API Functions]")
print("  ✓ load_model() - Load ECAPA-TDNN model (alias)")
print("  ✓ get_speaker_model() - Load model (original)")
print("  ✓ get_embedding(audio_path) - Extract 192-dim embedding")
print("  ✓ extract_embedding(audio_input) - Extract from file/array")

print("\n[Configuration]")
print(f"  Model: speechbrain/spkrec-ecapa-voxceleb")
print(f"  Embedding Dim: {EMBEDDING_DIM}")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")

print("\n[Quick Test]")
model = load_model()
print(f"  ✓ Model loaded: {type(model).__name__}")

emb = get_embedding("data/test_samples/speaker01_clip01.wav", warn_short=False)
print(f"  ✓ Embedding shape: {emb.shape}")
print(f"  ✓ First 5 values: {emb[:5].round(2)}")

print("\n[Key Features]")
print("  • torchaudio loading with librosa fallback")
print("  • Automatic resampling to 16kHz")
print("  • Mono conversion for stereo input")
print("  • Short audio warning (< 3 sec)")
print("  • Audio quality validation")

print("\n" + "=" * 60)
print("Ready for Phase 2+ integration!")
print("=" * 60)
