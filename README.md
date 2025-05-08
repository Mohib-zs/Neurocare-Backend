# NeuroCare Backend

A comprehensive mental health analysis API that provides text, audio, and video analysis for mental well-being assessment.

## Project Structure

```
NeuroCare-Backend/
├── .env                    # Environment variables
├── main.py                 # Main FastAPI application
├── models.py               # Database models
├── schemas.py              # Pydantic schemas
├── database.py             # Database configuration
├── security.py             # Security utilities
├── auth_utils.py           # Authentication utilities
├── email_utils.py          # Email utilities
├── emotion_utils.py        # Emotion analysis utilities
├── text_analysis_utils.py  # Text analysis utilities
├── audio_analysis_utils.py # Audio analysis utilities
├── video_emotion_utils.py  # Video analysis utilities
├── realtime_utils.py       # Real-time processing utilities
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Features

- **Text Analysis**: Analyze journal entries and thoughts for sentiment, emotions, and mental health indicators
- **Audio Analysis**: Process voice recordings for emotional state and speech patterns
- **Video Analysis**: Analyze facial expressions and voice for comprehensive emotional assessment
- **Trend Analysis**: Track emotional and mental health trends over time
- **Personalized Interventions**: Generate customized mental health recommendations
- **Real-time Processing**: Fast and efficient analysis of user inputs
- **Secure Authentication**: OAuth2 with JWT token-based authentication
- **Email Notifications**: Automated email notifications for analysis results
- **Scalable Architecture**: Built with FastAPI and PostgreSQL

## Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **Database**: PostgreSQL
- **Authentication**: JWT with OAuth2
- **ML Models**: Hugging Face Transformers
- **Email**: SMTP (Gmail)
- **Frontend Integration**: React (localhost:5173)

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 13 or higher
- Git
- FFmpeg (for audio/video processing)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Tahadevv/NeuroCare-Backend.git
cd NeuroCare-Backend
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory with:
```env
# Database Configuration
DATABASE_URL=postgresql://postgres:taha@localhost:5432/fastapi_db

# Security Settings
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email Settings
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=iamtahakhan3@gmail.com
SMTP_PASSWORD=wzlg ykkp ycqo efxh
FROM_EMAIL=iamtahakhan3@gmail.com
FRONTEND_URL=http://localhost:5173

# API Keys
HUGGINGFACE_API_KEY=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
HUGGINGFACE_TOKEN=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
```

5. **Initialize the database**
```bash
# Create database
createdb fastapi_db

# Run migrations
alembic upgrade head
```

6. **Start the server**
```bash
uvicorn main:app --reload
```

## API Endpoints

### Authentication

#### Sign Up
- **URL**: `/signup`
- **Method**: `POST`
- **Headers**: 
  - `Content-Type: application/json`
- **Body**:
```json
{
    "email": "user@example.com",
    "full_name": "John Doe",
    "password": "securepassword123"
}
```

#### Login
- **URL**: `/login`
- **Method**: `POST`
- **Headers**: 
  - `Content-Type: application/json`
- **Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword123"
}
```

### Text Analysis

#### Analyze Text
- **URL**: `/mental-health/analyze/text`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
  - `Content-Type: application/json`
- **Body**:
```json
{
    "content": "I've been feeling stressed at work lately. The deadlines are piling up and I'm having trouble sleeping.",
    "context": "Evening reflection"
}
```

### Audio Analysis

#### Analyze Audio
- **URL**: `/mental-health/analyze/audio`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Body**: 
  - `audio`: (binary) WAV file
  - `context`: (optional) string

### Video Analysis

#### Analyze Video
- **URL**: `/mental-health/analyze/video`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Body**: 
  - `video`: (binary) MP4 file
  - `context`: (optional) string

### Trend Analysis

#### Get Emotion Trends
- **URL**: `/mental-health/trends/emotions/{days}`
- **Method**: `GET`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Parameters**:
  - `days`: number of days (1-365)

#### Get Mental Health Trends
- **URL**: `/mental-health/trends/mental-health/{days}`
- **Method**: `GET`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Parameters**:
  - `days`: number of days (1-365)

### User Profile

#### Create Profile
- **URL**: `/user/profile`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
  - `Content-Type: application/json`
- **Body**:
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

#### Get Profile
- **URL**: `/user/profile`
- **Method**: `GET`
- **Headers**: 
  - `Authorization: Bearer <access_token>`

## Example Responses

### Text Analysis Response
```json
{
    "analysis_id": 1,
    "timestamp": "2024-03-15T10:30:00Z",
    "sentiment_analysis": {
        "sentiment": "NEGATIVE",
        "score": 0.85,
        "emotions": {
            "stress": 0.8,
            "anxiety": 0.6
        }
    },
    "recommendations": [
        "Practice deep breathing exercises",
        "Try time management techniques"
    ]
}
```

### Audio Analysis Response
```json
{
    "session_id": "uuid-string",
    "timestamp": "2024-03-15T10:30:00Z",
    "emotion_analysis": {
        "dominant_emotion": "stressed",
        "confidence": 0.82,
        "voice_characteristics": {
            "pitch": "higher than normal",
            "speech_rate": "faster than normal"
        }
    },
    "recommendations": [
        "Try progressive muscle relaxation",
        "Practice mindfulness meditation"
    ]
}
```

### Video Analysis Response
```json
{
    "facial_analysis": {
        "emotions": {
            "stress": 0.7,
            "tension": 0.6
        }
    },
    "recommendations": [
        "Take regular breaks during work",
        "Practice facial relaxation exercises"
    ]
}
```

## Error Responses

### Authentication Error
```json
{
    "detail": "Invalid credentials",
    "code": "AUTH_ERROR"
}
```

### Validation Error
```json
{
    "detail": [
        {
            "loc": ["body", "email"],
            "msg": "Invalid email format",
            "type": "value_error"
        }
    ]
}
```

### File Upload Error
```json
{
    "detail": "File size exceeds 10MB limit",
    "code": "FILE_TOO_LARGE"
}
```

## Email Notifications

The system sends email notifications for:
- Analysis results
- Account verification
- Password reset
- Important updates

## Frontend Integration

The backend is configured to work with a React frontend running on:
- URL: http://localhost:5173
- CORS enabled for development

## Support

For support, email iamtahakhan3@gmail.com or open an issue in the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Monitoring

The application includes:
- Prometheus metrics endpoint
- Sentry error tracking
- Structured logging
- Health check endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- FastAPI team for the amazing framework
- Hugging Face for the transformer models
- OpenCV for computer vision capabilities
- All contributors and maintainers 