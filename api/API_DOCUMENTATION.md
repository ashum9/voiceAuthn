# VoiceAuth API Documentation

## Base URL
```
http://localhost:8000
```

## API Prefix
All main endpoints use `/api/v1` prefix.

---

## Endpoints

### Health & Status

#### GET /health
Health check endpoint.

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

#### GET /stats
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

### Challenge Phrases

#### POST /api/v1/challenge
Get a random challenge phrase for authentication.

**Request Body (optional):**
```json
{
  "language": "english"  // Options: english, hindi, marathi
}
```

**Response:**
```json
{
  "challenge_id": "ch_abc123def456",
  "text": "The quick brown fox jumps over the lazy dog near the river bank every single morning",
  "language": "english",
  "language_display": "English",
  "expires_at": "2026-01-30T14:01:00.000000Z",
  "expires_in_seconds": 60
}
```

#### GET /api/v1/challenge?language=hindi
GET version of challenge creation.

#### GET /api/v1/challenge/languages
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

### Enrollment

#### POST /api/v1/enroll
Enroll user with single voice sample (quick enrollment).

**Request:** `multipart/form-data`
- `user_id` (string, required): Unique user identifier
- `audio` (file, required): Voice recording (WAV, WebM, MP3, OGG)
- `overwrite` (boolean, optional): Overwrite existing enrollment

**Response:**
```json
{
  "success": true,
  "message": "Voice enrollment completed successfully",
  "user_id": "user_123",
  "samples_count": 1,
  "quality_score": null,
  "consistency_score": null,
  "created_at": "2026-01-30T14:00:00.000000Z"
}
```

#### POST /api/v1/enroll/multi
Enroll user with multiple voice samples (3-5 recommended).

**Request:** `multipart/form-data`
- `user_id` (string, required)
- `audio_files` (file[], required): 1-5 voice recordings
- `overwrite` (boolean, optional)

#### GET /api/v1/enroll/check/{user_id}
Check if user is enrolled.

**Response:**
```json
{
  "exists": true,
  "user_id": "user_123"
}
```

---

### Authentication

#### POST /api/v1/auth
Authenticate user by voice.

**Request:** `multipart/form-data`
- `user_id` (string, required): User ID to authenticate against
- `audio` (file, required): Voice recording
- `challenge_id` (string, optional): Challenge ID for verification

**Response (Success):**
```json
{
  "success": true,
  "authenticated": true,
  "user_id": "user_123",
  "message": "Voice authentication successful",
  "similarity_score": null,
  "threshold": 0.5,
  "timestamp": "2026-01-30T14:00:00.000000Z"
}
```

**Response (Failure):**
```json
{
  "success": true,
  "authenticated": false,
  "user_id": "user_123",
  "message": "Voice authentication failed. Similarity: 35.2%",
  "similarity_score": 0.352,
  "threshold": 0.5,
  "timestamp": "2026-01-30T14:00:00.000000Z"
}
```

#### GET /api/v1/auth/threshold
Get authentication threshold.

**Response:**
```json
{
  "threshold": 0.5,
  "percentage": "50%",
  "description": "Voice recordings must match at least this percentage to authenticate successfully."
}
```

---

### User Management

#### GET /api/v1/users
List all enrolled users.

**Response:**
```json
{
  "users": ["user_123", "user_456"],
  "total_count": 2
}
```

#### GET /api/v1/users/{user_id}
Check if user exists.

#### DELETE /api/v1/users/{user_id}
Delete user and voiceprint.

**Response:**
```json
{
  "success": true,
  "message": "User and voiceprint deleted successfully",
  "user_id": "user_123"
}
```

---

## Error Responses

```json
{
  "success": false,
  "error": "User not found",
  "error_code": "USER_NOT_FOUND",
  "details": null,
  "timestamp": "2026-01-30T14:00:00.000000Z"
}
```

Common error codes:
- `400 Bad Request`: Invalid audio or user ID
- `404 Not Found`: User not found
- `409 Conflict`: User already enrolled
- `413 Request Entity Too Large`: Audio file too large

---

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## Running the API

```bash
cd /Users/ashum9/p1/voiceauth-mvp
/Users/ashum9/p1/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:
```bash
/Users/ashum9/p1/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --app-dir /Users/ashum9/p1/voiceauth-mvp
```
