# VoiceAuth Frontend Implementation Guide

## For: Frontend Development Team
## Project: VoiceAuth - Voice Authentication System
## Date: January 2026

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [API Configuration](#2-api-configuration)
3. [API Endpoints Reference](#3-api-endpoints-reference)
4. [User Flows](#4-user-flows)
5. [Audio Recording Requirements](#5-audio-recording-requirements)
6. [UI/UX Specifications](#6-uiux-specifications)
7. [Component Structure](#7-component-structure)
8. [State Management](#8-state-management)
9. [Error Handling](#9-error-handling)
10. [Security Considerations](#10-security-considerations)
11. [Testing Checklist](#11-testing-checklist)

---

# 1. Project Overview

## What is VoiceAuth?
A voice-based authentication system that uses AI (ECAPA-TDNN model) to verify users by their voice. Users enroll their voice once, then authenticate by speaking a challenge phrase.

## Key Features to Implement
- **Voice Enrollment**: Record user's voice (single or multiple samples)
- **Voice Authentication**: Verify user identity by voice
- **Challenge-Response**: Display phrases for users to speak
- **User Management**: List, check, delete users
- **Multilingual Support**: English, Hindi, Marathi phrases

## Technology Stack (Backend - Already Built)
- FastAPI (Python)
- ECAPA-TDNN (192-dim speaker embeddings)
- SQLite with encryption
- 50% similarity threshold

---

# 2. API Configuration

## Base URL
```
Development: http://localhost:8000
Production: [TO BE CONFIGURED]
```

## API Prefix
All main endpoints use `/api/v1` prefix.

## CORS
Already configured for:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)

## Headers
```javascript
// For JSON requests
headers: {
  'Content-Type': 'application/json'
}

// For file uploads (audio)
// Do NOT set Content-Type - let browser set multipart/form-data boundary
```

---

# 3. API Endpoints Reference

## 3.1 Health & Status

### GET /health
Check API status.

**Response:**
```json
{
  "status": "healthy",
  "message": "VoiceAuth API is running",
  "version": "1.0.0",
  "timestamp": "2026-01-30T14:00:00.000000Z",
  "model_loaded": true
}
```

### GET /stats
System statistics.

**Response:**
```json
{
  "total_users": 7,
  "database_mode": "sqlite",
  "database_path": "data/voiceauth.db",
  "model_loaded": true,
  "embedding_dim": 192
}
```

---

## 3.2 Challenge Phrases

### GET /api/v1/challenge?language={lang}
Get a random challenge phrase.

**Query Parameters:**
| Parameter | Type | Required | Values |
|-----------|------|----------|--------|
| language | string | No | `english`, `hindi`, `marathi` |

**Response:**
```json
{
  "challenge_id": "ch_abc123def456",
  "text": "The quick brown fox jumps over the lazy dog near the river bank",
  "language": "english",
  "language_display": "English",
  "expires_at": "2026-01-30T14:01:00.000000Z",
  "expires_in_seconds": 60
}
```

### POST /api/v1/challenge
Same as GET but with JSON body.

**Request Body:**
```json
{
  "language": "hindi"
}
```

### GET /api/v1/challenge/languages
List supported languages.

**Response:**
```json
{
  "languages": [
    {"code": "english", "display_name": "English", "phrase_count": 15},
    {"code": "hindi", "display_name": "हिंदी (Hindi)", "phrase_count": 15},
    {"code": "marathi", "display_name": "मराठी (Marathi)", "phrase_count": 15}
  ],
  "default": "english"
}
```

---

## 3.3 Enrollment

### POST /api/v1/enroll
Enroll user with single voice sample.

**Request:** `multipart/form-data`
```javascript
const formData = new FormData();
formData.append('user_id', 'john_doe');
formData.append('audio', audioBlob, 'recording.webm');
formData.append('overwrite', 'false'); // optional

fetch('/api/v1/enroll', {
  method: 'POST',
  body: formData
});
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Voice enrollment completed successfully",
  "user_id": "john_doe",
  "samples_count": 1,
  "quality_score": null,
  "consistency_score": null,
  "created_at": "2026-01-30T14:00:00.000000Z"
}
```

**Error Response (409 - User Exists):**
```json
{
  "detail": "User 'john_doe' is already enrolled. Set overwrite=true to re-enroll."
}
```

### POST /api/v1/enroll/multi
Enroll with multiple samples (recommended: 3 samples).

**Request:** `multipart/form-data`
```javascript
const formData = new FormData();
formData.append('user_id', 'john_doe');
formData.append('audio_files', audioBlob1, 'sample1.webm');
formData.append('audio_files', audioBlob2, 'sample2.webm');
formData.append('audio_files', audioBlob3, 'sample3.webm');
formData.append('overwrite', 'false');

fetch('/api/v1/enroll/multi', {
  method: 'POST',
  body: formData
});
```

**Response:**
```json
{
  "success": true,
  "message": "Voice enrollment completed successfully",
  "user_id": "john_doe",
  "samples_count": 3,
  "quality_score": 0.85,
  "consistency_score": 0.92,
  "created_at": "2026-01-30T14:00:00.000000Z"
}
```

### GET /api/v1/enroll/check/{user_id}
Check if user is enrolled.

**Response:**
```json
{
  "exists": true,
  "user_id": "john_doe"
}
```

---

## 3.4 Authentication

### POST /api/v1/auth
Authenticate user by voice.

**Request:** `multipart/form-data`
```javascript
const formData = new FormData();
formData.append('user_id', 'john_doe');
formData.append('audio', audioBlob, 'recording.webm');
formData.append('challenge_id', 'ch_abc123'); // optional

fetch('/api/v1/auth', {
  method: 'POST',
  body: formData
});
```

**Success Response (Authenticated):**
```json
{
  "success": true,
  "authenticated": true,
  "user_id": "john_doe",
  "message": "Voice authentication successful",
  "similarity_score": null,
  "threshold": 0.5,
  "timestamp": "2026-01-30T14:00:00.000000Z"
}
```

**Success Response (Not Authenticated):**
```json
{
  "success": true,
  "authenticated": false,
  "user_id": "john_doe",
  "message": "Voice authentication failed. Similarity: 35.2%",
  "similarity_score": 0.352,
  "threshold": 0.5,
  "timestamp": "2026-01-30T14:00:00.000000Z"
}
```

> **IMPORTANT**: On success, `similarity_score` is `null`. On failure, it shows the actual score.

### GET /api/v1/auth/threshold
Get authentication threshold.

**Response:**
```json
{
  "threshold": 0.5,
  "percentage": "50%",
  "description": "Voice recordings must match at least 50% to authenticate."
}
```

---

## 3.5 User Management

### GET /api/v1/users
List all enrolled users.

**Response:**
```json
{
  "users": ["john_doe", "jane_smith", "user123"],
  "total_count": 3
}
```

### GET /api/v1/users/{user_id}
Check if specific user exists.

**Response:**
```json
{
  "exists": true,
  "user_id": "john_doe"
}
```

### DELETE /api/v1/users/{user_id}
Delete user and their voiceprint.

**Response:**
```json
{
  "success": true,
  "message": "User and voiceprint deleted successfully",
  "user_id": "john_doe"
}
```

---

# 4. User Flows

## 4.1 Enrollment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENROLLMENT FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User enters User ID                                         │
│          ↓                                                      │
│  2. Check if user exists: GET /api/v1/enroll/check/{user_id}    │
│          ↓                                                      │
│  ┌───────┴───────┐                                              │
│  │ User Exists?  │                                              │
│  └───────┬───────┘                                              │
│      YES │        NO                                            │
│          ↓         ↓                                            │
│  Show "Already    3. Select enrollment type:                    │
│  enrolled" msg       - Quick (1 sample)                         │
│  + Re-enroll         - Multi (3 samples) ← RECOMMENDED          │
│  option                      ↓                                  │
│                      4. Get challenge phrase:                   │
│                         GET /api/v1/challenge?language=english  │
│                              ↓                                  │
│                      5. Display phrase to user                  │
│                              ↓                                  │
│                      6. User clicks "Record"                    │
│                              ↓                                  │
│                      7. Record audio (5-10 seconds)             │
│                              ↓                                  │
│                      8. [Multi only] Repeat steps 4-7           │
│                         for 2 more samples                      │
│                              ↓                                  │
│                      9. Submit: POST /api/v1/enroll             │
│                         or POST /api/v1/enroll/multi            │
│                              ↓                                  │
│                      10. Show success/error message             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTHENTICATION FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User enters User ID                                         │
│          ↓                                                      │
│  2. Check if user exists: GET /api/v1/users/{user_id}           │
│          ↓                                                      │
│  ┌───────┴───────┐                                              │
│  │ User Exists?  │                                              │
│  └───────┬───────┘                                              │
│      NO  │        YES                                           │
│          ↓         ↓                                            │
│  Show "Not        3. Select language (optional)                 │
│  enrolled" msg           ↓                                      │
│  + Enroll link    4. Get challenge phrase:                      │
│                      GET /api/v1/challenge?language=english     │
│                          ↓                                      │
│                   5. Display phrase + countdown timer (60s)     │
│                          ↓                                      │
│                   6. User clicks "Record"                       │
│                          ↓                                      │
│                   7. Record audio (5-10 seconds)                │
│                          ↓                                      │
│                   8. Submit: POST /api/v1/auth                  │
│                      (include challenge_id)                     │
│                          ↓                                      │
│                   ┌──────┴──────┐                               │
│                   │ authenticated│                              │
│                   └──────┬──────┘                               │
│                   true   │    false                             │
│                      ↓         ↓                                │
│              Show SUCCESS   Show FAILURE                        │
│              (green, no     (red, show                          │
│               score)        similarity %)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 5. Audio Recording Requirements

## 5.1 Browser Audio Recording

Use the MediaRecorder API:

```javascript
// Request microphone access
const stream = await navigator.mediaDevices.getUserMedia({ 
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    sampleRate: 16000  // Preferred, but browser may ignore
  } 
});

// Create recorder
const mediaRecorder = new MediaRecorder(stream, {
  mimeType: 'audio/webm;codecs=opus'  // Best browser support
});

// Collect audio chunks
const chunks = [];
mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

// Get final blob
mediaRecorder.onstop = () => {
  const audioBlob = new Blob(chunks, { type: 'audio/webm' });
  // Use audioBlob for upload
};

// Start recording
mediaRecorder.start();

// Stop after duration
setTimeout(() => mediaRecorder.stop(), 8000); // 8 seconds
```

## 5.2 Audio Specifications

| Property | Requirement |
|----------|-------------|
| **Format** | WebM, WAV, MP3, OGG (WebM recommended) |
| **Duration** | Minimum 3 seconds, Maximum 15 seconds |
| **Recommended** | 5-10 seconds of clear speech |
| **Sample Rate** | 16kHz (converted server-side) |
| **Channels** | Mono (converted server-side) |
| **Max File Size** | 10 MB |

## 5.3 Recording UI Requirements

1. **Visual Feedback**
   - Show recording indicator (red dot, pulsing animation)
   - Display elapsed time
   - Show audio waveform visualization (optional but recommended)

2. **Countdown/Timer**
   - Show remaining time for challenge phrase (60 seconds)
   - Show recording duration

3. **Controls**
   - Start/Stop recording button
   - Playback recorded audio before submit
   - Re-record option
   - Cancel option

## 5.4 Recommended Libraries

```javascript
// Option 1: react-media-recorder
npm install react-media-recorder

// Option 2: react-audio-recorder
npm install react-audio-recorder

// Option 3: Custom with MediaRecorder API (shown above)
```

---

# 6. UI/UX Specifications

## 6.1 Design Guidelines

### Color Scheme (Recommended)
```css
/* Primary Colors */
--primary: #1a1a2e;      /* Dark blue-black */
--secondary: #16213e;    /* Navy blue */
--accent: #0f3460;       /* Deep blue */
--success: #22c55e;      /* Green */
--error: #ef4444;        /* Red */
--warning: #f59e0b;      /* Amber */

/* Text Colors */
--text-primary: #ffffff;
--text-secondary: #94a3b8;

/* Background */
--bg-primary: #0f0f1a;
--bg-secondary: #1a1a2e;
```

### Typography
- Modern, clean sans-serif fonts
- Clear hierarchy for phrases (large, readable)
- Support for Hindi and Marathi scripts (Devanagari)

## 6.2 Pages Required

### 1. Home/Landing Page
- Brief explanation of voice authentication
- Navigation to Enroll and Authenticate
- System status indicator

### 2. Enrollment Page
- User ID input
- Enrollment type selection (Quick/Multi)
- Language selection dropdown
- Recording interface
- Progress indicator for multi-sample
- Success/Error feedback

### 3. Authentication Page
- User ID input
- Language selection
- Challenge phrase display (LARGE, PROMINENT)
- Countdown timer
- Recording interface
- Result display:
  - SUCCESS: Green checkmark, "Authenticated" message
  - FAILURE: Red X, "Failed" message with similarity percentage

### 4. User Management Page (Admin)
- List of enrolled users
- Delete user functionality
- User count display

### 5. Settings Page (Optional)
- Language preferences
- Audio device selection
- Theme toggle

## 6.3 Challenge Phrase Display

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   Please speak the following phrase:                       │
│                                                            │
│   ┌──────────────────────────────────────────────────────┐ │
│   │                                                      │ │
│   │  "The quick brown fox jumps over the lazy dog       │ │
│   │   near the river bank every single morning"         │ │
│   │                                                      │ │
│   └──────────────────────────────────────────────────────┘ │
│                                                            │
│   Language: English                    Expires in: 45s     │
│                                                            │
│                    [ 🎤 Start Recording ]                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 6.4 Result Display

### Success State
```
┌────────────────────────────────────────┐
│                                        │
│            ✓ AUTHENTICATED             │
│                                        │
│     Voice verification successful      │
│                                        │
│         Welcome, john_doe!             │
│                                        │
└────────────────────────────────────────┘
```

### Failure State
```
┌────────────────────────────────────────┐
│                                        │
│         ✗ AUTHENTICATION FAILED        │
│                                        │
│      Voice did not match profile       │
│                                        │
│         Similarity: 35.2%              │
│         Required: 50%                  │
│                                        │
│            [ Try Again ]               │
│                                        │
└────────────────────────────────────────┘
```

---

# 7. Component Structure

## 7.1 Recommended React Component Tree

```
src/
├── components/
│   ├── common/
│   │   ├── Button.jsx
│   │   ├── Input.jsx
│   │   ├── Card.jsx
│   │   ├── Modal.jsx
│   │   ├── Spinner.jsx
│   │   └── Alert.jsx
│   │
│   ├── audio/
│   │   ├── AudioRecorder.jsx      # Main recording component
│   │   ├── AudioWaveform.jsx      # Visual waveform display
│   │   ├── AudioPlayback.jsx      # Playback recorded audio
│   │   └── RecordingTimer.jsx     # Recording duration display
│   │
│   ├── challenge/
│   │   ├── ChallengeDisplay.jsx   # Display phrase to speak
│   │   ├── LanguageSelector.jsx   # Language dropdown
│   │   └── ChallengeTimer.jsx     # 60s countdown timer
│   │
│   ├── enrollment/
│   │   ├── EnrollmentForm.jsx     # User ID + type selection
│   │   ├── EnrollmentProgress.jsx # Multi-sample progress
│   │   └── EnrollmentResult.jsx   # Success/error display
│   │
│   ├── auth/
│   │   ├── AuthForm.jsx           # User ID input
│   │   ├── AuthRecording.jsx      # Recording for auth
│   │   └── AuthResult.jsx         # Auth result display
│   │
│   └── users/
│       ├── UserList.jsx           # List of users
│       └── UserCard.jsx           # Individual user card
│
├── pages/
│   ├── HomePage.jsx
│   ├── EnrollPage.jsx
│   ├── AuthPage.jsx
│   ├── UsersPage.jsx
│   └── SettingsPage.jsx
│
├── hooks/
│   ├── useAudioRecorder.js        # Audio recording logic
│   ├── useChallenge.js            # Challenge phrase fetching
│   ├── useEnrollment.js           # Enrollment API calls
│   ├── useAuth.js                 # Authentication API calls
│   └── useUsers.js                # User management API calls
│
├── services/
│   └── api.js                     # API client configuration
│
├── utils/
│   ├── audioUtils.js              # Audio processing helpers
│   └── formatters.js              # Date, percentage formatters
│
└── App.jsx
```

## 7.2 Key Component Specifications

### AudioRecorder Component
```jsx
<AudioRecorder
  onRecordingComplete={(audioBlob) => handleAudio(audioBlob)}
  maxDuration={15}           // seconds
  minDuration={3}            // seconds
  showWaveform={true}
  showTimer={true}
  onError={(error) => handleError(error)}
/>
```

### ChallengeDisplay Component
```jsx
<ChallengeDisplay
  text="The quick brown fox..."
  language="english"
  languageDisplay="English"
  expiresAt={timestamp}
  onExpired={() => fetchNewChallenge()}
/>
```

### AuthResult Component
```jsx
<AuthResult
  authenticated={true}
  userId="john_doe"
  similarityScore={null}     // null on success
  threshold={0.5}
  onRetry={() => resetAuth()}
/>
```

---

# 8. State Management

## 8.1 Recommended: React Context or Zustand

### Global State Structure
```javascript
{
  // API Status
  api: {
    isConnected: true,
    modelLoaded: true,
    error: null
  },
  
  // Current User Session
  session: {
    userId: null,
    isEnrolled: false,
    isAuthenticated: false
  },
  
  // Challenge State
  challenge: {
    id: null,
    text: null,
    language: 'english',
    expiresAt: null,
    isLoading: false
  },
  
  // Recording State
  recording: {
    isRecording: false,
    audioBlob: null,
    duration: 0,
    error: null
  },
  
  // UI Preferences
  preferences: {
    language: 'english',
    theme: 'dark'
  }
}
```

## 8.2 API Service Layer

```javascript
// services/api.js

const API_BASE = 'http://localhost:8000';
const API_PREFIX = '/api/v1';

export const api = {
  // Health
  getHealth: () => fetch(`${API_BASE}/health`).then(r => r.json()),
  getStats: () => fetch(`${API_BASE}/stats`).then(r => r.json()),
  
  // Challenge
  getChallenge: (language) => 
    fetch(`${API_BASE}${API_PREFIX}/challenge?language=${language || ''}`).then(r => r.json()),
  
  getLanguages: () => 
    fetch(`${API_BASE}${API_PREFIX}/challenge/languages`).then(r => r.json()),
  
  // Enrollment
  checkEnrollment: (userId) => 
    fetch(`${API_BASE}${API_PREFIX}/enroll/check/${userId}`).then(r => r.json()),
  
  enroll: (userId, audioBlob, overwrite = false) => {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('overwrite', overwrite.toString());
    return fetch(`${API_BASE}${API_PREFIX}/enroll`, {
      method: 'POST',
      body: formData
    }).then(r => r.json());
  },
  
  enrollMulti: (userId, audioBlobs, overwrite = false) => {
    const formData = new FormData();
    formData.append('user_id', userId);
    audioBlobs.forEach((blob, i) => {
      formData.append('audio_files', blob, `sample${i + 1}.webm`);
    });
    formData.append('overwrite', overwrite.toString());
    return fetch(`${API_BASE}${API_PREFIX}/enroll/multi`, {
      method: 'POST',
      body: formData
    }).then(r => r.json());
  },
  
  // Authentication
  authenticate: (userId, audioBlob, challengeId = null) => {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('audio', audioBlob, 'recording.webm');
    if (challengeId) formData.append('challenge_id', challengeId);
    return fetch(`${API_BASE}${API_PREFIX}/auth`, {
      method: 'POST',
      body: formData
    }).then(r => r.json());
  },
  
  // Users
  getUsers: () => 
    fetch(`${API_BASE}${API_PREFIX}/users`).then(r => r.json()),
  
  deleteUser: (userId) => 
    fetch(`${API_BASE}${API_PREFIX}/users/${userId}`, { method: 'DELETE' }).then(r => r.json()),
};
```

---

# 9. Error Handling

## 9.1 HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Show validation error |
| 404 | Not Found | User not enrolled |
| 409 | Conflict | User already exists |
| 413 | Too Large | Audio file too big |
| 500 | Server Error | Show generic error |
| 503 | Unavailable | ML model loading |

## 9.2 Error Response Format

```json
{
  "detail": "Error message here"
}
```

Or for validation errors:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "audio"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

## 9.3 Error Messages to Display

| Error | User Message |
|-------|--------------|
| User not found | "User not enrolled. Please enroll first." |
| User already enrolled | "User already enrolled. Would you like to re-enroll?" |
| Audio too short | "Recording too short. Please speak for at least 3 seconds." |
| Audio too long | "Recording too long. Maximum 15 seconds allowed." |
| Challenge expired | "Challenge expired. Getting a new phrase..." |
| Network error | "Connection failed. Please check your internet." |
| Microphone denied | "Microphone access required. Please allow access." |

---

# 10. Security Considerations

## 10.1 User ID Validation
- Allow only: letters, numbers, underscores, hyphens
- Length: 1-100 characters
- Regex: `^[\w\-]+$`

```javascript
const isValidUserId = (userId) => /^[\w\-]+$/.test(userId);
```

## 10.2 Audio Handling
- Never store audio files on frontend
- Clear audio blobs after upload
- Use HTTPS in production

## 10.3 Challenge Security
- Challenges expire in 60 seconds
- Each challenge can only be used once
- Display countdown timer to user

---

# 11. Testing Checklist

## 11.1 Functional Testing

### Enrollment
- [ ] Single sample enrollment works
- [ ] Multi-sample enrollment works (3 samples)
- [ ] Duplicate user shows appropriate error
- [ ] Overwrite enrollment works
- [ ] Invalid user ID is rejected

### Authentication
- [ ] Successful authentication shows green success
- [ ] Failed authentication shows red with percentage
- [ ] Non-existent user shows appropriate error
- [ ] Challenge countdown works
- [ ] New challenge after expiry

### Audio Recording
- [ ] Microphone permission prompt appears
- [ ] Recording indicator visible
- [ ] Timer counts correctly
- [ ] Audio playback works
- [ ] Re-record option works

### User Management
- [ ] User list loads correctly
- [ ] Delete user works
- [ ] Deleted user cannot authenticate

## 11.2 Edge Cases

- [ ] Very short audio (< 3s) rejected
- [ ] Very long audio (> 15s) rejected
- [ ] Poor network handling
- [ ] Microphone access denied
- [ ] Browser compatibility (Chrome, Firefox, Safari, Edge)
- [ ] Mobile browser compatibility

## 11.3 Accessibility

- [ ] Keyboard navigation
- [ ] Screen reader support
- [ ] Color contrast
- [ ] Focus indicators
- [ ] Error announcements

---

# Appendix A: Quick Start Code

## Minimal React Setup

```bash
# Create project
npm create vite@latest voiceauth-frontend -- --template react

# Install dependencies
cd voiceauth-frontend
npm install axios

# Start development
npm run dev
```

## Minimal API Test

```javascript
// Test API connection
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(data => console.log('API Status:', data.status))
  .catch(err => console.error('API Error:', err));
```

---

# Appendix B: API Swagger UI

Interactive API documentation available at:
```
http://localhost:8000/docs
```

Use this to test all endpoints directly in the browser.

---

# Contact & Support

For API issues or questions:
- Check API logs: `/tmp/api.log`
- Swagger docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`

---

*Document Version: 1.0*
*Last Updated: January 30, 2026*
