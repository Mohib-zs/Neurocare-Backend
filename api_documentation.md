# Mental Health Analysis API Documentation

## Base URL
```
https://your-domain.com/api/v1
```

## API Versioning
- Current Version: v1
- Version Format: v{MAJOR}.{MINOR}
- Version Header: `X-API-Version: v1`
- Backward Compatibility: Maintained for 6 months after new version release
- Deprecation Notice: Sent 3 months before version removal

## Authentication
The API uses OAuth2 with Bearer token authentication.

### Authentication Headers
```
Authorization: Bearer <access_token>
```

## Endpoints

### Authentication & User Management

#### 1. User Signup
- **URL**: `/signup`
- **Method**: `POST`
- **Auth Required**: No
- **Request Body**:
```json
{
    "email": "user@example.com",
    "full_name": "John Doe",
    "password": "securepassword123"
}
```
- **Response**: User object
```json
{
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "is_active": true,
    "created_at": "2024-03-15T10:30:00Z",
    "updated_at": null
}
```

#### 2. User Login
- **URL**: `/login`
- **Method**: `POST`
- **Auth Required**: No
- **Request Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword123"
}
```
- **Response**: Access token
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer"
}
```

#### 3. Password Reset Request
- **URL**: `/reset-password`
- **Method**: `POST`
- **Auth Required**: No
- **Request Body**:
```json
{
    "email": "user@example.com"
}
```
- **Response**:
```json
{
    "message": "Password reset OTP sent to your email",
    "otp": "123456"  // Only in development
}
```

### User Profile Management

#### 1. Create User Profile
- **URL**: `/user/profile`
- **Method**: `POST`
- **Auth Required**: Yes
- **Request Body**:
```json
{
    "name": "John Doe",
    "age": 30,
    "gender": "Male",
    "sleep_hours_actual": 7.5,
    "sleep_hours_target": 8.0,
    "goals": [
        "Reduce Stress & Anxiety",
        "Improve Sleep Quality"
    ]
}
```
- **Response**: User profile object

#### 2. Get User Profile
- **URL**: `/user/profile`
- **Method**: `GET`
- **Auth Required**: Yes
- **Response**: User profile object

### Mental Health Analysis

#### 1. Text Analysis
- **URL**: `/mental-health/analyze/text`
- **Method**: `POST`
- **Auth Required**: Yes
- **Request Body**:
```json
{
    "content": "Your journal entry or thoughts here...",
    "context": "Evening reflection"
}
```
- **Response**:
```json
{
    "analysis_id": 1,
    "timestamp": "2024-03-15T10:30:00Z",
    "sentiment_analysis": {
        "sentiment": {
            "label": "POSITIVE",
            "score": 0.85,
            "polarity": 0.75,
            "subjectivity": 0.6
        },
        "emotions": {
            "label": "joy",
            "score": 0.82
        }
    },
    "linguistic_analysis": {
        "sentence_analysis": {
            "avg_length": 15.5,
            "max_length": 25,
            "num_sentences": 10
        },
        "linguistic_features": {
            "question_marks": 2,
            "exclamation_marks": 1,
            "ellipsis": 0,
            "negative_words": 3,
            "personal_pronouns": 8
        },
        "vocabulary_diversity": 0.75,
        "word_count": 150
    },
    "themes_analysis": {
        "main_themes": ["work stress", "family", "personal growth"],
        "potential_concerns": {
            "anxiety": 2,
            "stress": 3,
            "relationships": 1
        }
    },
    "mental_health_assessment": {
        "risk_level": 3,
        "areas_of_concern": ["work-life balance", "stress management"],
        "emotional_state": {
            "primary_emotion": "anxious",
            "intensity": 0.6,
            "stability": "moderate"
        },
        "coping_strategies": [
            "Deep breathing exercises",
            "Time management techniques"
        ],
        "professional_help": {
            "recommended": false,
            "type": ["Counseling"],
            "urgency": "low"
        },
        "immediate_support": [
            "Practice mindfulness",
            "Talk to a friend"
        ]
    },
    "personalized_interventions": {
        "daily_practices": [
            "Morning meditation",
            "Evening journaling"
        ],
        "weekly_goals": [
            "Schedule two relaxation periods",
            "Connect with one friend"
        ],
        "crisis_plan": {
            "immediate_steps": [
                "Deep breathing",
                "Call support person"
            ],
            "emergency_contacts": [
                "Therapist",
                "Crisis Hotline"
            ]
        },
        "reflection_prompts": [
            "What triggered my stress today?",
            "What helped me feel calm?"
        ],
        "progress_metrics": [
            "Daily stress level",
            "Sleep quality"
        ]
    }
}
```

#### 2. Audio Analysis
- **URL**: `/mental-health/analyze/audio`
- **Method**: `POST`
- **Auth Required**: Yes
- **Request Body**: Form Data
  - `audio`: WAV file
- **Response**:
```json
{
    "session_id": "uuid-string",
    "timestamp": "2024-03-15T10:30:00Z",
    "audio_features": {
        "pitch_mean": 120.5,
        "pitch_std": 15.2,
        "energy": 0.75,
        "tempo": 95.0,
        "speech_rate": 150.0,
        "pause_ratio": 0.2,
        "voice_quality": 0.85
    },
    "emotion_analysis": {
        "arousal": 0.65,
        "valence": 0.45,
        "dominant_emotion": "calm",
        "confidence": 0.82,
        "emotion_scores": {
            "calm": 0.82,
            "anxious": 0.12,
            "stressed": 0.06
        }
    },
    "speech_content": {
        "transcription": "Transcribed text here...",
        "sentiment_score": 0.65,
        "key_phrases": ["feeling better", "making progress"],
        "hesitation_count": 3,
        "word_per_minute": 150.0
    },
    "mental_state": {
        "stress_level": 0.4,
        "anxiety_level": 0.3,
        "depression_indicators": ["low energy"],
        "mood_state": "stable",
        "energy_level": 0.6,
        "coherence_score": 0.75,
        "emotional_stability": 0.7,
        "sleep_quality_indicator": 0.65,
        "social_engagement_level": 0.8,
        "cognitive_load": 0.45,
        "resilience_score": 0.75
    },
    "mental_health_scores": {
        "anxiety_score": 0.3,
        "depression_score": 0.2,
        "stress_score": 0.4,
        "emotional_regulation": 0.7,
        "social_connection": 0.8,
        "mindfulness": 0.6,
        "sleep_quality": 0.65,
        "cognitive_performance": 0.75,
        "resilience": 0.7,
        "life_satisfaction": 0.75
    },
    "recommendations": [
        "Practice deep breathing exercises",
        "Maintain regular sleep schedule"
    ],
    "follow_up_questions": [
        "How has your sleep been lately?",
        "What activities help you relax?"
    ],
    "intervention_plan": {
        "short_term": [
            "Morning meditation",
            "Evening relaxation"
        ],
        "long_term": [
            "Build support network",
            "Develop coping strategies"
        ]
    }
}
```

#### 3. Video Analysis
- **URL**: `/mental-health/analyze/video`
- **Method**: `POST`
- **Auth Required**: Yes
- **Request Body**: Form Data
  - `video`: Video file
- **Response**:
```json
{
    "facial_analysis": {
        "average_emotions": {
            "happy": 0.6,
            "neutral": 0.3,
            "sad": 0.1
        },
        "dominant_emotion": "happy"
    },
    "voice_analysis": {
        "arousal": 0.7,
        "valence": 0.65
    },
    "speech_analysis": {
        "transcription": "Transcribed text here...",
        "sentiment": {
            "positive": 0.75,
            "negative": 0.25
        }
    },
    "overall_assessment": {
        "emotional_state": "positive",
        "confidence_score": 0.85
    },
    "intervention": "Personalized intervention text..."
}
```

### Trend Analysis

#### 1. Emotion Trends
- **URL**: `/mental-health/trends/emotions/{days}`
- **Method**: `GET`
- **Auth Required**: Yes
- **Parameters**:
  - `days`: Number of days (1-365)
- **Response**:
```json
{
    "daily_frequencies": {
        "2024-03-15": {
            "happy": 5,
            "calm": 3
        }
    },
    "dominant_emotions": {
        "happy": 0.45,
        "calm": 0.35
    },
    "emotion_stability": 0.75,
    "mood_variability": 0.3,
    "positive_ratio": 0.8
}
```

#### 2. Mental Health Trends
- **URL**: `/mental-health/trends/mental-health/{days}`
- **Method**: `GET`
- **Auth Required**: Yes
- **Parameters**:
  - `days`: Number of days (1-365)
- **Response**:
```json
{
    "daily_metrics": {
        "2024-03-15": {
            "stress_level": 0.4,
            "anxiety_level": 0.3,
            "mood_state": "stable",
            "emotional_stability": 0.7,
            "social_engagement_level": 0.8,
            "cognitive_load": 0.45,
            "resilience_score": 0.75
        }
    },
    "overall_trends": {
        "stress_level": [0.4, 0.35, 0.3],
        "anxiety_level": [0.3, 0.25, 0.2]
    },
    "risk_factors": {
        "sleep_disruption": 2,
        "high_stress": 1
    },
    "improvement_areas": [
        "stress_management",
        "sleep_quality"
    ],
    "stability_metrics": {
        "emotional_stability": {
            "average": 0.7,
            "variance": 0.1,
            "trend": "improving"
        }
    }
}
```

## Models and Schemas

### User Models
```python
class UserCreate:
    email: str
    full_name: str
    password: str

class UserProfile:
    name: str
    age: int
    gender: Gender  # Enum: Male, Female, Non-binary, Prefer not to say
    sleep_hours_actual: float
    sleep_hours_target: float
    goals: List[Goal]  # List of predefined goals
```

### Analysis Models
```python
class TextAnalysis:
    content: str
    context: Optional[str]

class AudioAnalysis:
    audio_features: AudioFeatures
    emotion_analysis: AudioEmotionAnalysis
    speech_content: SpeechContent
    mental_state: MentalStateIndicators
    mental_health_scores: MentalHealthScores
```

## Error Responses

### Common Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 413: Payload Too Large
- 429: Too Many Requests
- 500: Internal Server Error
- 502: Bad Gateway
- 503: Service Unavailable
- 504: Gateway Timeout

### Error Response Format
```json
{
    "detail": "Error message description",
    "code": "ERROR_CODE",
    "timestamp": "2024-03-15T10:30:00Z"
}
```

### Specific Error Codes
- `FILE_TOO_LARGE`: File size exceeds 10MB limit
- `INVALID_FILE_FORMAT`: Unsupported file format
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `DB_CONNECTION_ERROR`: Database connection failed
- `API_KEY_INVALID`: Invalid API key
- `MODEL_LOAD_ERROR`: Failed to load ML model

## Rate Limiting
- Rate limit: 100 requests per minute per user
- Rate limit headers included in response:
  - X-RateLimit-Limit
  - X-RateLimit-Remaining
  - X-RateLimit-Reset
- Burst limit: 20 requests per 10 seconds
- Global rate limit: 1000 requests per minute

## File Upload Specifications
- Maximum file size: 10MB
- Supported audio formats: WAV, MP3, M4A
- Supported video formats: MP4, MOV, AVI
- Maximum video duration: 5 minutes
- Maximum audio duration: 10 minutes

## Pagination
- Default page size: 20 items
- Maximum page size: 100 items
- Pagination parameters:
  - `page`: Page number (default: 1)
  - `size`: Items per page (default: 20)
  - `sort`: Sort field (default: timestamp)
  - `order`: Sort order (asc/desc)

## Environment Variables Required
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/db_name
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys
HUGGINGFACE_API_KEY=your-huggingface-api-key
OPENAI_API_KEY=your-openai-api-key

# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=your-aws-region
S3_BUCKET_NAME=your-bucket-name

# Application Settings
MAX_FILE_SIZE=10485760  # 10MB in bytes
MAX_VIDEO_DURATION=300  # 5 minutes in seconds
MAX_AUDIO_DURATION=600  # 10 minutes in seconds
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

## Dependencies
```requirements.txt
# Web Framework & Server
fastapi==0.109.2
uvicorn==0.27.1
python-multipart==0.0.9
httpx==0.26.0
starlette==0.36.3
pydantic==2.6.1
pydantic-settings==2.1.0
email-validator==2.1.0.post1

# Database & ORM
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
alembic==1.13.1
redis==5.0.1
aioredis==2.0.1
asyncpg==0.29.0

# Authentication & Security
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.1.2
python-dotenv==1.0.1
cryptography==42.0.2
itsdangerous==2.1.2

# Data Processing & Analysis
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
scipy==1.12.0
matplotlib==3.8.3
seaborn==0.13.2

# Machine Learning & NLP
transformers==4.37.2
torch==2.2.0
torchaudio==2.2.0
torchvision==0.17.0
spacy==3.7.2
textblob==0.17.1
nltk==3.8.1
sentence-transformers==2.5.1
tokenizers==0.15.2

# Mistral Model
accelerate==0.27.2
bitsandbytes==0.42.0
safetensors==0.4.1
einops==0.7.0
flash-attn==2.5.5
xformers==0.0.23.post1

# Audio Processing
librosa==0.10.1
pydub==0.25.1
soundfile==0.12.1
pyaudio==0.2.14
audioread==3.0.1
resampy==0.4.2

# Video Processing
opencv-python==4.9.0.80
moviepy==1.0.3
ffmpeg-python==0.2.0
imageio==2.34.0
pillow==10.2.0

# AWS & Cloud Services
boto3==1.34.34
botocore==1.34.34
s3transfer==0.10.1
aiobotocore==2.11.1

# Testing & Development
pytest==8.0.1
pytest-asyncio==0.23.5
pytest-cov==4.1.0
black==24.1.1
isort==5.13.2
flake8==7.0.0
mypy==1.8.0

# Monitoring & Logging
prometheus-client==0.19.0
sentry-sdk==1.39.1
structlog==24.1.0
python-json-logger==2.0.7

# Utilities
python-dateutil==2.8.2
pytz==2024.1
tzlocal==5.2
requests==2.31.0
aiohttp==3.9.3
tenacity==8.2.3
tqdm==4.66.1
rich==13.7.0
```

## Database Models
```python
# User Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    sleep_hours_actual = Column(Float)
    sleep_hours_target = Column(Float)
    goals = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

# Analysis Models
class TextAnalysis(Base):
    __tablename__ = "text_analyses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    context = Column(String, nullable=True)
    sentiment_score = Column(Float)
    emotion_scores = Column(JSON)
    linguistic_metrics = Column(JSON)
    identified_themes = Column(JSON)
    concerns = Column(JSON)
    risk_level = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class AudioAnalysis(Base):
    __tablename__ = "audio_analyses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, unique=True)
    audio_features = Column(JSON)
    emotion_analysis = Column(JSON)
    speech_content = Column(JSON)
    mental_state = Column(JSON)
    mental_health_scores = Column(JSON)
    recommendations = Column(JSON)
    follow_up_questions = Column(JSON)
    intervention_plan = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class VideoAnalysis(Base):
    __tablename__ = "video_analyses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    facial_analysis = Column(JSON)
    voice_analysis = Column(JSON)
    speech_analysis = Column(JSON)
    overall_assessment = Column(JSON)
    intervention = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Trend Analysis Models
class EmotionTrend(Base):
    __tablename__ = "emotion_trends"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date)
    emotion_frequencies = Column(JSON)
    dominant_emotions = Column(JSON)
    emotion_stability = Column(Float)
    mood_variability = Column(Float)
    positive_ratio = Column(Float)

class MentalHealthTrend(Base):
    __tablename__ = "mental_health_trends"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date)
    daily_metrics = Column(JSON)
    overall_trends = Column(JSON)
    risk_factors = Column(JSON)
    improvement_areas = Column(JSON)
    stability_metrics = Column(JSON)
```

## Additional System Requirements
```bash
# Required System Packages
apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    build-essential \
    libpq-dev \
    redis-server \
    postgresql \
    postgresql-contrib

# Required Python Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Post-installation Steps
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader wordnet
```

## Deployment Notes
1. Use HTTPS in production
2. Set up proper CORS configuration
3. Configure proper logging
4. Set up monitoring and alerting
5. Configure proper backup strategy for the database
6. Use proper security headers
7. Set up proper error tracking
8. Configure Redis for caching and rate limiting
9. Set up AWS S3 for file storage
10. Configure proper load balancing
11. Set up auto-scaling
12. Configure proper SSL/TLS certificates
13. Set up proper firewall rules
14. Configure proper database replication
15. Set up proper backup and disaster recovery procedures 