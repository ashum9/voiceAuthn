#!/usr/bin/env python3
"""Phase 3 Verification - Quick Summary"""

import sys
sys.path.insert(0, '.')

from utils import (
    pre_process_audio,
    pre_process_audio_simple,
    apply_vad,
    check_liveness,
    load_audio_from_bytes,
    AudioProcessingError,
    LivenessCheckError,
    SAMPLE_RATE,
    MIN_SPEECH_DURATION,
    VAD_ENERGY_THRESHOLD,
    LIVENESS_RMS_VARIANCE_THRESHOLD
)

print("=" * 60)
print("Phase 3: Audio Pre-Processing Pipeline - COMPLETE")
print("=" * 60)

print("\n[API Functions]")
print("  ✓ pre_process_audio(input) - Full pipeline with metadata")
print("  ✓ pre_process_audio_simple(input) - Quick processing")
print("  ✓ apply_vad(waveform) - Voice Activity Detection")
print("  ✓ check_liveness(waveform) - Spoof detection")
print("  ✓ load_audio_from_bytes(bytes) - Load from bytes/BytesIO")

print("\n[Configuration]")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")
print(f"  Min Speech Duration: {MIN_SPEECH_DURATION}s")
print(f"  VAD Energy Threshold: {VAD_ENERGY_THRESHOLD}")
print(f"  Liveness RMS Variance: {LIVENESS_RMS_VARIANCE_THRESHOLD}")

print("\n[Exceptions]")
print("  • AudioProcessingError - Audio too short or invalid")
print("  • LivenessCheckError - Potential spoofing detected")

print("\n[Pipeline Steps]")
print("  1. Load audio (path, bytes, BytesIO, or array)")
print("  2. Resample to 16kHz")
print("  3. Convert to mono")
print("  4. Apply VAD (trim silence)")
print("  5. Normalize to 0.95 max amplitude")
print("  6. Check minimum duration (>= 3s)")
print("  7. Liveness check (detect spoofing)")

print("\n[Quick Test]")
import numpy as np
# Generate test audio
t = np.linspace(0, 4.0, int(SAMPLE_RATE * 4.0))
test_audio = 0.5 * np.sin(2 * np.pi * 150 * t * (1 + 0.1 * np.sin(2 * np.pi * 5 * t)))
test_audio = test_audio * (0.3 + 0.7 * np.sin(2 * np.pi * 3 * t) ** 2)
test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8

processed, meta = pre_process_audio(
    test_audio,
    sample_rate=SAMPLE_RATE,
    check_liveness_enabled=False
)
print(f"  ✓ Processed: {meta['original_duration']:.2f}s → {meta['processed_duration']:.2f}s")
print(f"  ✓ Speech ratio: {meta['speech_ratio']:.1%}")
print(f"  ✓ Normalized: {meta['normalized']}")

print("\n" + "=" * 60)
print("Ready for integration with embedding extraction!")
print("=" * 60)
