"""
VoiceAuth API - Audio Service
=============================
Audio processing utilities for the API.

Handles:
- Audio format conversion (WebM → WAV)
- Saving temporary files
- Audio validation and preprocessing
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

import numpy as np

# Try to import pydub for format conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("[AudioService] Warning: pydub not installed. WebM conversion may fail.")

# Import from main project
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    load_audio_from_bytes,
    preprocess_audio,
    validate_audio_quality,
    SAMPLE_RATE,
)

from ..config import TEMP_DIR, MIN_AUDIO_DURATION, MAX_AUDIO_DURATION


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


def convert_audio_to_wav(audio_bytes: bytes, source_format: str = "webm") -> bytes:
    """
    Convert audio bytes to WAV format.
    
    Browsers typically record in WebM or OGG format.
    This converts to WAV for processing with our ML pipeline.
    
    Args:
        audio_bytes: Raw audio data
        source_format: Source format (webm, ogg, mp3, etc.)
        
    Returns:
        WAV audio bytes
        
    Raises:
        AudioProcessingError: If conversion fails
    """
    if not PYDUB_AVAILABLE:
        # Try to process directly without conversion
        return audio_bytes
    
    try:
        # Load audio with pydub
        audio_io = io.BytesIO(audio_bytes)
        
        # Try to detect format automatically, fallback to specified
        try:
            audio = AudioSegment.from_file(audio_io, format=source_format)
        except Exception:
            # Try auto-detection
            audio_io.seek(0)
            audio = AudioSegment.from_file(audio_io)
        
        # Convert to mono, 16kHz, 16-bit WAV
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(SAMPLE_RATE)
        audio = audio.set_sample_width(2)  # 16-bit
        
        # Export to WAV
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        return wav_io.read()
        
    except Exception as e:
        raise AudioProcessingError(f"Failed to convert audio: {str(e)}")


def process_audio_bytes(
    audio_bytes: bytes,
    convert_format: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Process raw audio bytes into a normalized waveform.
    
    Steps:
    1. Convert to WAV if needed
    2. Load as numpy array
    3. Resample to 16kHz
    4. Validate quality
    
    Args:
        audio_bytes: Raw audio data
        convert_format: Whether to attempt format conversion
        
    Returns:
        Tuple of (waveform as numpy array, metadata dict)
        
    Raises:
        AudioProcessingError: If processing fails
    """
    metadata = {
        "original_size": len(audio_bytes),
        "converted": False,
        "duration": 0,
        "sample_rate": SAMPLE_RATE,
    }
    
    # Step 1: Try to convert format if needed
    if convert_format and PYDUB_AVAILABLE:
        try:
            audio_bytes = convert_audio_to_wav(audio_bytes)
            metadata["converted"] = True
        except Exception:
            # Continue with original bytes
            pass
    
    # Step 2: Load audio
    try:
        waveform, sr = load_audio_from_bytes(audio_bytes)
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    
    # Step 3: Preprocess (resample, normalize, etc.)
    try:
        waveform = preprocess_audio(waveform, sr, target_sr=SAMPLE_RATE)
    except Exception as e:
        raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
    
    # Calculate duration
    duration = len(waveform) / SAMPLE_RATE
    metadata["duration"] = duration
    
    # Step 4: Validate quality
    is_valid, message = validate_audio_quality(
        waveform,
        sample_rate=SAMPLE_RATE,
        min_duration=MIN_AUDIO_DURATION,
        max_duration=MAX_AUDIO_DURATION
    )
    
    if not is_valid:
        raise AudioProcessingError(message)
    
    metadata["quality_message"] = message
    
    return waveform, metadata


def save_temp_audio(audio_bytes: bytes, prefix: str = "upload") -> str:
    """
    Save audio bytes to a temporary file.
    
    Args:
        audio_bytes: Raw audio data
        prefix: Filename prefix
        
    Returns:
        Path to the temporary file
        
    Note:
        Caller is responsible for deleting the file after use.
    """
    # Ensure temp directory exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.wav"
    filepath = TEMP_DIR / filename
    
    # Write to file
    with open(filepath, 'wb') as f:
        f.write(audio_bytes)
    
    return str(filepath)


def cleanup_temp_file(filepath: str) -> bool:
    """
    Delete a temporary audio file.
    
    Args:
        filepath: Path to the file to delete
        
    Returns:
        True if deleted, False if failed or not found
    """
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            return True
    except Exception as e:
        print(f"[AudioService] Warning: Failed to delete temp file {filepath}: {e}")
    return False


def get_audio_duration(audio_bytes: bytes) -> Optional[float]:
    """
    Get the duration of audio in seconds.
    
    Args:
        audio_bytes: Raw audio data
        
    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        if PYDUB_AVAILABLE:
            audio_io = io.BytesIO(audio_bytes)
            audio = AudioSegment.from_file(audio_io)
            return len(audio) / 1000.0  # Convert ms to seconds
        else:
            waveform, sr = load_audio_from_bytes(audio_bytes)
            return len(waveform) / sr
    except Exception:
        return None
