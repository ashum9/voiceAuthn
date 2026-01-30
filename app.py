"""
VoiceAuth MVP - Streamlit Application
=====================================
Voice-Based Authentication Demo with:
- User Enrollment (record voice, extract embedding, store securely)
- User Authentication (record voice, compare to stored voiceprint)
- Challenge-Response System (anti-replay protection)
- Live Microphone Recording (Phase 5.1)

Requires: streamlit, audio-recorder-streamlit, numpy
Run with: streamlit run app.py --server.port 8501
"""

import streamlit as st
import numpy as np
import tempfile
import os
import io
from datetime import datetime, timezone
import random

# Audio recorder component
from audio_recorder_streamlit import audio_recorder

# Import our modules
from utils import (
    get_embedding,
    get_speaker_model,
    compute_similarity,
    generate_challenge,
    get_random_challenge,
    SIMILARITY_THRESHOLD,
    DEMO_THRESHOLD,
    EMBEDDING_DIM,
    SAMPLE_RATE,
    pre_process_audio,
    AudioProcessingError,
    LivenessCheckError,
    PHRASES_ENGLISH,
    LANGUAGE_NAMES,
    save_audio_temp,
    cleanup_temp_file,
    average_embeddings,
    compute_enrollment_quality,
    normalize_embedding,
)
from storage import (
    get_voiceprint_db,
    VoiceprintDB,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="VoiceAuth MVP",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .phrase-box {
        padding: 1.5rem;
        background-color: #FFF3E0;
        border: 2px dashed #FF9800;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .recording-area {
        padding: 1.5rem;
        background-color: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        color: #333333 !important;
    }
    .recording-area * {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'db' not in st.session_state:
    st.session_state.db = get_voiceprint_db()

if 'current_phrase' not in st.session_state:
    st.session_state.current_phrase = None

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None

# Multi-enrollment session state
if 'enroll_embeddings' not in st.session_state:
    st.session_state.enroll_embeddings = []

if 'enroll_phrases' not in st.session_state:
    st.session_state.enroll_phrases = []

if 'enroll_step' not in st.session_state:
    st.session_state.enroll_step = 0

if 'enroll_user_id' not in st.session_state:
    st.session_state.enroll_user_id = ""

# Default number of enrollments
NUM_ENROLLMENTS = 3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_cached():
    """Load and cache the speaker model."""
    return get_speaker_model()


def process_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, str]:
    """
    Process audio bytes from microphone and extract embedding.
    
    Returns:
        Tuple of (embedding, status_message)
    """
    if not audio_bytes or len(audio_bytes) < 1000:
        return None, "No audio recorded or recording too short"
    
    # Save to temp file
    tmp_path = save_audio_temp(audio_bytes, prefix="recording")
    
    try:
        # Pre-process audio (VAD, normalization)
        try:
            processed_audio, metadata = pre_process_audio(
                tmp_path,
                apply_vad_trimming=True,
                check_liveness_enabled=False,  # Disabled for demo
                normalize_audio=True
            )
            
            # Save processed audio to temp file for embedding extraction
            import soundfile as sf
            processed_path = tmp_path.replace('.wav', '_processed.wav')
            sf.write(processed_path, processed_audio, SAMPLE_RATE)
            
        except AudioProcessingError as e:
            return None, f"Audio processing error: {e}"
        except LivenessCheckError as e:
            return None, f"Liveness check failed: {e}"
        
        # Extract embedding
        embedding = get_embedding(processed_path, warn_short=False)
        
        # Clean up processed file
        cleanup_temp_file(processed_path)
        
        return embedding, "Success"
        
    except Exception as e:
        return None, f"Error extracting embedding: {e}"
    finally:
        # Clean up temp file
        cleanup_temp_file(tmp_path)


def process_audio_file(audio_file) -> tuple[np.ndarray, str]:
    """
    Process uploaded audio file and extract embedding.
    
    Returns:
        Tuple of (embedding, status_message)
    """
    return process_audio_bytes(audio_file.getvalue())


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎤 VoiceAuth MVP")
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["🏠 Home", "📝 Enroll", "🔐 Authenticate", "👥 Users", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model status
    try:
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                load_model_cached()
                st.session_state.model_loaded = True
        st.success("✅ Model loaded")
    except Exception as e:
        st.error(f"❌ Model error: {e}")
    
    # Database stats
    st.markdown("### 📊 Database")
    stats = st.session_state.db.get_stats()
    st.metric("Enrolled Users", stats.get('users', 0))
    st.caption(f"Mode: {stats.get('mode', 'unknown')}")
    
    st.markdown("---")
    st.caption("VoiceAuth MVP v1.1")
    st.caption("Phase 5.1: Live Recording")

# ============================================================================
# HOME PAGE
# ============================================================================

if mode == "🏠 Home":
    st.markdown('<div class="main-header">🎤 VoiceAuth MVP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Voice-Based Authentication System</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Enroll")
        st.markdown("Register your voice by recording a sample phrase.")
        
    with col2:
        st.markdown("### 🔐 Authenticate")
        st.markdown("Verify your identity using your unique voiceprint.")
        
    with col3:
        st.markdown("### 🔒 Secure")
        st.markdown("Encrypted storage with GDPR compliance.")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Go to Enroll** - Create a new user and record your voice
    2. **Go to Authenticate** - Test voice verification
    3. **Check Users** - View enrolled users
    """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        - **Model**: ECAPA-TDNN (SpeechBrain)
        - **Embedding Dimension**: {EMBEDDING_DIM}
        - **Sample Rate**: {SAMPLE_RATE} Hz
        - **Similarity Threshold**: {SIMILARITY_THRESHOLD}
        - **Demo Threshold**: {DEMO_THRESHOLD}
        - **Encryption**: Fernet (AES-128-CBC + HMAC)
        - **Recording**: Browser MediaRecorder API
        """)

# ============================================================================
# ENROLLMENT PAGE (Phase 6: Multi-Enrollment with Averaging)
# ============================================================================

elif mode == "📝 Enroll":
    st.markdown("## 📝 Voice Enrollment")
    st.markdown("Create a robust voice profile by recording **multiple samples**.")
    
    st.markdown("---")
    
    # Enrollment mode selection
    enroll_mode = st.radio(
        "Enrollment Mode",
        ["🎯 Multi-Sample (Recommended)", "⚡ Quick Single Sample"],
        horizontal=True,
        help="Multi-sample creates a more robust voiceprint by averaging 3 recordings"
    )
    
    st.markdown("---")
    
    # User ID input
    user_id = st.text_input(
        "User ID",
        placeholder="Enter a unique username",
        help="This will be your identifier for authentication",
        key="enroll_user_input"
    )
    
    # Sync user ID to session state
    if user_id:
        st.session_state.enroll_user_id = user_id
    
    # Check if user exists
    if user_id:
        if st.session_state.db.exists(user_id):
            st.warning(f"⚠️ User '{user_id}' already exists. Enrolling will update their voiceprint.")
    
    # ========== MULTI-SAMPLE ENROLLMENT ==========
    if enroll_mode == "🎯 Multi-Sample (Recommended)":
        st.markdown(f"### 📊 Progress: {len(st.session_state.enroll_embeddings)}/{NUM_ENROLLMENTS} samples")
        
        # Progress bar
        progress = len(st.session_state.enroll_embeddings) / NUM_ENROLLMENTS
        st.progress(progress)
        
        # Show collected samples
        if st.session_state.enroll_embeddings:
            with st.expander(f"📋 Collected {len(st.session_state.enroll_embeddings)} samples"):
                for i, phrase in enumerate(st.session_state.enroll_phrases):
                    st.write(f"✅ Sample {i+1}: \"{phrase[:50]}...\"" if len(phrase) > 50 else f"✅ Sample {i+1}: \"{phrase}\"")
        
        # If not all samples collected, show recording interface
        if len(st.session_state.enroll_embeddings) < NUM_ENROLLMENTS:
            current_sample = len(st.session_state.enroll_embeddings) + 1
            
            st.markdown(f"### 🎙️ Recording Sample {current_sample} of {NUM_ENROLLMENTS}")
            
            # Initialize enrollment language if not set
            if 'enroll_language' not in st.session_state:
                st.session_state.enroll_language = "random"
            if 'enroll_challenge' not in st.session_state:
                st.session_state.enroll_challenge = None
            
            # Language selection for enrollment
            lang_col1, lang_col2 = st.columns([2, 1])
            with lang_col1:
                language_options = ["random", "english", "hindi", "marathi"]
                language_labels = {
                    "random": "🎲 Random Language",
                    "english": "🇬🇧 English",
                    "hindi": "🇮🇳 हिंदी (Hindi)",
                    "marathi": "🇮🇳 मराठी (Marathi)"
                }
                selected_lang = st.selectbox(
                    "Select Language",
                    options=language_options,
                    format_func=lambda x: language_labels[x],
                    index=language_options.index(st.session_state.enroll_language),
                    key="enroll_lang_select"
                )
                if selected_lang != st.session_state.enroll_language:
                    st.session_state.enroll_language = selected_lang
            
            with lang_col2:
                if st.button("🔄 New Phrase", use_container_width=True, key="multi_enroll_phrase"):
                    lang = None if st.session_state.enroll_language == "random" else st.session_state.enroll_language
                    st.session_state.enroll_challenge = get_random_challenge(lang)
                    st.rerun()
            
            # Display the challenge phrase
            if st.session_state.enroll_challenge:
                challenge = st.session_state.enroll_challenge
                st.markdown(f'''
                <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); 
                            color: white; padding: 20px; border-radius: 12px; 
                            font-size: 16px; text-align: center; margin: 10px 0;">
                    <div style="font-size: 11px; opacity: 0.9; margin-bottom: 6px;">🌐 {challenge['lang_display']}</div>
                    📢 "{challenge['text']}"
                </div>
                ''', unsafe_allow_html=True)
                st.markdown('<p style="text-align: center; color: #666;">⏱️ Speak naturally – 8-12 seconds</p>', unsafe_allow_html=True)
            else:
                st.info("👆 Click 'New Phrase' to get a phrase to speak")
            
            st.markdown("---")
            
            # Audio input
            st.markdown('<div class="recording-area">', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #333; font-weight: bold;">Record Sample {current_sample}: Click microphone and speak the phrase (8-12 seconds)</p>', unsafe_allow_html=True)
            
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#1E88E5",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=3.0,
                sample_rate=16000,
                key=f"multi_enroll_rec_{current_sample}"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.success(f"✅ Recording captured: {len(audio_bytes):,} bytes")
                
                # Process and add button
                if st.button(f"➕ Add Sample {current_sample}", type="primary", use_container_width=True):
                    if not user_id:
                        st.error("Please enter a User ID first")
                    else:
                        with st.spinner(f"Processing sample {current_sample}..."):
                            embedding, status = process_audio_bytes(audio_bytes)
                            
                            if embedding is not None:
                                st.session_state.enroll_embeddings.append(embedding)
                                phrase_text = st.session_state.enroll_challenge['text'] if st.session_state.enroll_challenge else "No phrase"
                                st.session_state.enroll_phrases.append(phrase_text)
                                # Auto-generate new phrase for next sample
                                lang = None if st.session_state.enroll_language == "random" else st.session_state.enroll_language
                                st.session_state.enroll_challenge = get_random_challenge(lang)
                                st.success(f"✅ Sample {current_sample} added!")
                                st.rerun()
                            else:
                                st.error(f"Failed to process: {status}")
        
        # All samples collected - show averaging option
        else:
            st.markdown("### ✅ All samples collected!")
            
            # Show quality metrics
            quality = compute_enrollment_quality(st.session_state.enroll_embeddings)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Consistency", f"{quality['consistency']:.2%}")
            with col2:
                st.metric("Min Similarity", f"{quality['min_similarity']:.2%}")
            with col3:
                st.metric("Samples", quality['num_samples'])
            
            if quality['consistency'] < 0.75:
                st.warning("⚠️ Low consistency between samples. Consider re-recording for better results.")
            else:
                st.success("✅ Good consistency! Ready to create voiceprint.")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Create Averaged Voiceprint", type="primary", use_container_width=True):
                    if not user_id:
                        st.error("Please enter a User ID")
                    else:
                        with st.spinner("Creating averaged voiceprint..."):
                            # Average embeddings
                            avg_embedding = average_embeddings(st.session_state.enroll_embeddings)
                            
                            # Store in database
                            success = st.session_state.db.store(user_id, avg_embedding)
                            
                            if success:
                                st.markdown(f'''
                                <div class="success-box">
                                    ✅ <strong>Multi-Sample Enrollment Complete!</strong><br>
                                    User "{user_id}" enrolled with {NUM_ENROLLMENTS} averaged samples.<br>
                                    Voiceprint consistency: {quality['consistency']:.2%}
                                </div>
                                ''', unsafe_allow_html=True)
                                st.balloons()
                                
                                # Clear session
                                st.session_state.enroll_embeddings = []
                                st.session_state.enroll_phrases = []
                                
                                with st.expander("📊 Averaged Embedding Details"):
                                    st.write(f"Dimension: {len(avg_embedding)}")
                                    st.write(f"Norm: {np.linalg.norm(avg_embedding):.4f}")
                                    st.write(f"Samples averaged: {quality['num_samples']}")
                                    st.write(f"Consistency score: {quality['consistency']:.4f}")
                            else:
                                st.error("Failed to store voiceprint")
            
            with col2:
                if st.button("🔄 Start Over", use_container_width=True):
                    st.session_state.enroll_embeddings = []
                    st.session_state.enroll_phrases = []
                    st.session_state.enroll_challenge = None
                    st.rerun()
        
        # Reset button (always visible during multi-enroll)
        if st.session_state.enroll_embeddings and len(st.session_state.enroll_embeddings) < NUM_ENROLLMENTS:
            st.markdown("---")
            if st.button("🗑️ Reset Enrollment", use_container_width=True):
                st.session_state.enroll_embeddings = []
                st.session_state.enroll_phrases = []
                st.rerun()
    
    # ========== QUICK SINGLE SAMPLE ENROLLMENT ==========
    else:
        st.info("⚡ Quick mode: Single sample enrollment (less robust than multi-sample)")
        
        # Initialize quick enrollment challenge
        if 'quick_enroll_language' not in st.session_state:
            st.session_state.quick_enroll_language = "random"
        if 'quick_enroll_challenge' not in st.session_state:
            st.session_state.quick_enroll_challenge = None
        
        # Generate challenge phrase with language selection
        st.markdown("### 📢 Challenge Phrase")
        lang_col1, lang_col2 = st.columns([2, 1])
        with lang_col1:
            language_options = ["random", "english", "hindi", "marathi"]
            language_labels = {
                "random": "🎲 Random Language",
                "english": "🇬🇧 English",
                "hindi": "🇮🇳 हिंदी (Hindi)",
                "marathi": "🇮🇳 मराठी (Marathi)"
            }
            selected_lang = st.selectbox(
                "Select Language",
                options=language_options,
                format_func=lambda x: language_labels[x],
                index=language_options.index(st.session_state.quick_enroll_language),
                key="quick_enroll_lang_select"
            )
            if selected_lang != st.session_state.quick_enroll_language:
                st.session_state.quick_enroll_language = selected_lang
        
        with lang_col2:
            if st.button("🔄 New Phrase", use_container_width=True, key="quick_enroll_phrase"):
                lang = None if st.session_state.quick_enroll_language == "random" else st.session_state.quick_enroll_language
                st.session_state.quick_enroll_challenge = get_random_challenge(lang)
                st.rerun()
        
        # Display the challenge phrase
        if st.session_state.quick_enroll_challenge:
            challenge = st.session_state.quick_enroll_challenge
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); 
                        color: white; padding: 20px; border-radius: 12px; 
                        font-size: 16px; text-align: center; margin: 10px 0;">
                <div style="font-size: 11px; opacity: 0.9; margin-bottom: 6px;">🌐 {challenge['lang_display']}</div>
                📢 "{challenge['text']}"
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; color: #666;">⏱️ Speak naturally – 8-12 seconds</p>', unsafe_allow_html=True)
        else:
            st.info("👆 Click 'New Phrase' to get a phrase to speak")
        
        st.markdown("---")
        
        # Audio input
        st.markdown("### 🎵 Record or Upload Voice Sample")
        input_method = st.radio(
            "Choose input method:",
            ["🎙️ Record with Microphone", "📁 Upload Audio File"],
            horizontal=True,
            label_visibility="collapsed",
            key="quick_input_method"
        )
        
        audio_bytes = None
        
        if input_method == "🎙️ Record with Microphone":
            st.markdown('<div class="recording-area">', unsafe_allow_html=True)
            st.markdown('<p style="color: #333; font-weight: bold;">Click the microphone to start recording (8-12 seconds)</p>', unsafe_allow_html=True)
            
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#1E88E5",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=3.0,
                sample_rate=16000,
                key="quick_enroll_recorder"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.success(f"✅ Recording captured: {len(audio_bytes):,} bytes")
        
        else:
            audio_file = st.file_uploader(
                "Upload a WAV or MP3 file (5-10 seconds)",
                type=["wav", "mp3", "ogg", "flac"],
                help="Record yourself speaking the phrase above",
                key="quick_enroll_upload"
            )
            if audio_file:
                audio_bytes = audio_file.getvalue()
                st.audio(audio_bytes, format="audio/wav")
        
        st.markdown("---")
        
        # Enrollment button
        if st.button("✅ Quick Enroll", type="primary", use_container_width=True):
            if not user_id:
                st.error("Please enter a User ID")
            elif not audio_bytes:
                st.error("Please record or upload audio first")
            else:
                with st.spinner("Processing voice sample..."):
                    embedding, status = process_audio_bytes(audio_bytes)
                    
                    if embedding is not None:
                        success = st.session_state.db.store(user_id, embedding)
                        
                        if success:
                            st.markdown(f'''
                            <div class="success-box">
                                ✅ <strong>Quick Enrollment Complete!</strong><br>
                                User "{user_id}" has been enrolled (single sample).
                            </div>
                            ''', unsafe_allow_html=True)
                            st.balloons()
                            
                            with st.expander("📊 Embedding Details"):
                                st.write(f"Dimension: {len(embedding)}")
                                st.write(f"Norm: {np.linalg.norm(embedding):.4f}")
                        else:
                            st.error("Failed to store voiceprint")
                    else:
                        st.error(status)

# ============================================================================
# AUTHENTICATION PAGE
# ============================================================================

elif mode == "🔐 Authenticate":
    st.markdown("## 🔐 Voice Authentication")
    st.markdown("Verify your identity using your voice.")
    
    st.markdown("---")
    
    # User ID input
    user_id = st.text_input(
        "User ID",
        placeholder="Enter your username",
        help="Enter the username you enrolled with"
    )
    
    # Check if user exists
    if user_id and not st.session_state.db.exists(user_id):
        st.warning(f"⚠️ User '{user_id}' not found. Please enroll first.")
    
    # Initialize auth challenge in session state
    if 'auth_challenge' not in st.session_state:
        st.session_state.auth_challenge = None
    if 'auth_language' not in st.session_state:
        st.session_state.auth_language = "random"
    
    # Challenge phrase section
    st.markdown("### 📝 Challenge Phrase")
    st.markdown("Read this phrase aloud to verify your identity:")
    
    # Language selection
    lang_col1, lang_col2 = st.columns([2, 1])
    with lang_col1:
        language_options = ["random", "english", "hindi", "marathi"]
        language_labels = {
            "random": "🎲 Random Language",
            "english": "🇬🇧 English",
            "hindi": "🇮🇳 हिंदी (Hindi)",
            "marathi": "🇮🇳 मराठी (Marathi)"
        }
        selected_lang = st.selectbox(
            "Select Language",
            options=language_options,
            format_func=lambda x: language_labels[x],
            index=language_options.index(st.session_state.auth_language),
            key="auth_lang_select"
        )
        if selected_lang != st.session_state.auth_language:
            st.session_state.auth_language = selected_lang
    
    with lang_col2:
        if st.button("🔄 Get New Phrase", use_container_width=True, key="auth_phrase"):
            lang = None if st.session_state.auth_language == "random" else st.session_state.auth_language
            st.session_state.auth_challenge = get_random_challenge(lang)
            st.rerun()
    
    # Display the challenge phrase
    if st.session_state.auth_challenge:
        challenge = st.session_state.auth_challenge
        lang_display = challenge['lang_display']
        
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 25px; border-radius: 15px; 
                    font-size: 18px; text-align: center; margin: 10px 0;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
            <div style="font-size: 12px; opacity: 0.9; margin-bottom: 8px;">
                🌐 {lang_display}
            </div>
            🎤 "{challenge['text']}"
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <p style="text-align: center; color: #888; margin-top: 10px;">
            ⏱️ Speak naturally, take your time – 8-12 seconds recommended
        </p>
        ''', unsafe_allow_html=True)
    else:
        st.info("👆 Click 'Get New Phrase' to receive a challenge phrase")
    
    st.markdown("---")
    
    # Audio input method selection
    st.markdown("### 🎵 Record or Upload Voice Sample")
    input_method = st.radio(
        "Choose input method:",
        ["🎙️ Record with Microphone", "📁 Upload Audio File"],
        horizontal=True,
        label_visibility="collapsed",
        key="auth_input_method"
    )
    
    audio_bytes = None
    
    if input_method == "🎙️ Record with Microphone":
        st.markdown('<div class="recording-area">', unsafe_allow_html=True)
        st.markdown('<p style="color: #333; font-weight: bold;">Click the microphone to start recording (8-12 seconds)</p>', unsafe_allow_html=True)
        
        # Audio recorder component
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#1E88E5",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=3.0,
            sample_rate=16000,
            key="auth_recorder"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.success(f"✅ Recording captured: {len(audio_bytes):,} bytes")
    
    else:
        # File upload fallback
        audio_file = st.file_uploader(
            "Upload a WAV or MP3 file",
            type=["wav", "mp3", "ogg", "flac"],
            help="Record yourself speaking the challenge phrase",
            key="auth_audio"
        )
        if audio_file:
            audio_bytes = audio_file.getvalue()
            st.audio(audio_bytes, format="audio/wav")
    
    st.markdown("---")
    
    # Fixed threshold at 50%
    threshold = 0.50
    
    # Authentication button
    if st.button("🔓 Authenticate", type="primary", use_container_width=True):
        if not user_id:
            st.error("Please enter a User ID")
        elif not st.session_state.db.exists(user_id):
            st.error(f"User '{user_id}' not found. Please enroll first.")
        elif not audio_bytes:
            st.error("Please record or upload audio first")
        else:
            with st.spinner("Verifying voice..."):
                # Extract embedding from audio
                test_embedding, status = process_audio_bytes(audio_bytes)
                
                if test_embedding is not None:
                    # Retrieve stored embedding
                    stored_embedding = st.session_state.db.retrieve(user_id)
                    
                    if stored_embedding is not None:
                        # Compute similarity
                        similarity = compute_similarity(test_embedding, stored_embedding)
                        is_match = similarity >= threshold
                        
                        # Display result - Phase 7: Clean UI
                        # Success: Show only message (no score)
                        # Failure: Show similarity score
                        if is_match:
                            st.markdown('''
                            <div style="background: linear-gradient(135deg, #2ecc71, #27ae60);
                                        color: white; padding: 40px; border-radius: 20px;
                                        text-align: center; margin: 20px 0;
                                        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.4);">
                                <div style="font-size: 48px; margin-bottom: 10px;">✅</div>
                                <div style="font-size: 28px; font-weight: bold;">Authentication Successful</div>
                            </div>
                            ''', unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f'''
                            <div style="background: linear-gradient(135deg, #e74c3c, #c0392b);
                                        color: white; padding: 40px; border-radius: 20px;
                                        text-align: center; margin: 20px 0;
                                        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.4);">
                                <div style="font-size: 48px; margin-bottom: 10px;">❌</div>
                                <div style="font-size: 28px; font-weight: bold;">Authentication Failed</div>
                                <div style="font-size: 18px; margin-top: 15px; opacity: 0.9;">
                                    Similarity: {similarity:.2%}
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.error("Failed to retrieve stored voiceprint")
                else:
                    st.error(status)

# ============================================================================
# USERS PAGE
# ============================================================================

elif mode == "👥 Users":
    st.markdown("## 👥 Enrolled Users")
    st.markdown("View and manage enrolled users.")
    
    st.markdown("---")
    
    users = st.session_state.db.list_users()
    
    if not users:
        st.info("No users enrolled yet. Go to Enroll to add users.")
    else:
        st.markdown(f"**{len(users)} users enrolled**")
        
        for user_id in users:
            with st.expander(f"👤 {user_id}"):
                embedding = st.session_state.db.retrieve(user_id)
                if embedding is not None:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Embedding Dimension:** {len(embedding)}")
                        st.write(f"**Norm:** {np.linalg.norm(embedding):.4f}")
                        st.write(f"**Mean:** {np.mean(embedding):.4f}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{user_id}"):
                            st.session_state.db.delete(user_id)
                            st.success(f"Deleted {user_id}")
                            st.rerun()
    
    st.markdown("---")
    
    # Database stats
    st.markdown("### 📊 Database Statistics")
    stats = st.session_state.db.get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", stats.get('users', 0))
    with col2:
        st.metric("Storage Mode", stats.get('mode', 'unknown').upper())
    with col3:
        st.metric("Encryption", "✅ Enabled")

# ============================================================================
# SETTINGS PAGE
# ============================================================================

elif mode == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    st.markdown("Configure system parameters.")
    
    st.markdown("---")
    
    st.markdown("### 🔧 Thresholds")
    st.info(f"""
    - **Standard Threshold:** {SIMILARITY_THRESHOLD}
    - **Demo Threshold:** {DEMO_THRESHOLD}
    - **Strict Threshold:** 0.85
    - **Lenient Threshold:** 0.70
    """)
    
    st.markdown("---")
    
    st.markdown("### 📦 Model Info")
    st.info(f"""
    - **Model:** ECAPA-TDNN
    - **Source:** speechbrain/spkrec-ecapa-voxceleb
    - **Embedding Dimension:** {EMBEDDING_DIM}
    - **Sample Rate:** {SAMPLE_RATE} Hz
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎙️ Recording Info")
    st.info("""
    - **Component:** audio-recorder-streamlit
    - **Format:** WAV (16kHz)
    - **API:** Browser MediaRecorder
    - **Privacy:** Audio deleted after processing
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔒 Security")
    st.info("""
    - **Encryption:** Fernet (AES-128-CBC + HMAC)
    - **Key Storage:** data/.encryption_key
    - **GDPR Compliant:** Yes (deletion supported)
    - **Audio Storage:** None (transient only)
    """)
    
    st.markdown("---")
    
    st.markdown("### 🧹 Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Temp Files", use_container_width=True):
            temp_dir = "data/temp"
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                st.success("Temp files cleared")
            else:
                st.info("No temp files to clear")
    
    with col2:
        confirm = st.checkbox("I confirm deletion of all users")
        if st.button("🗑️ Clear All Users", type="secondary", use_container_width=True):
            if confirm:
                for uid in st.session_state.db.list_users():
                    st.session_state.db.delete(uid)
                st.success("All users deleted")
                st.rerun()
            else:
                st.warning("Please confirm deletion first")
    
    st.markdown("---")
    
    st.markdown("### 📋 System Status")
    
    # Check all components
    checks = []
    
    # Model check
    try:
        model = load_model_cached()
        checks.append(("Model", "✅ Loaded"))
    except:
        checks.append(("Model", "❌ Error"))
    
    # Database check
    try:
        stats = st.session_state.db.get_stats()
        checks.append(("Database", f"✅ {stats.get('mode', 'unknown').upper()}"))
    except:
        checks.append(("Database", "❌ Error"))
    
    # Encryption check
    if os.path.exists("data/.encryption_key"):
        checks.append(("Encryption Key", "✅ Found"))
    else:
        checks.append(("Encryption Key", "⚠️ Will be generated"))
    
    # Temp directory
    if os.path.exists("data/temp"):
        temp_files = len(os.listdir("data/temp"))
        checks.append(("Temp Directory", f"✅ {temp_files} files"))
    else:
        checks.append(("Temp Directory", "📁 Will be created"))
    
    for name, status in checks:
        st.write(f"**{name}:** {status}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("VoiceAuth MVP v1.1 | Built with SpeechBrain & Streamlit | Phase 5.1: Live Recording")