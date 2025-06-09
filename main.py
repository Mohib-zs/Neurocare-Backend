# Standard library imports
import os
import json
import logging
import uuid
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional

# Third-party imports
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Body, Path
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from sqlalchemy.orm import Session
from sqlalchemy import func
from dotenv import load_dotenv
try:
    import jwt
except ImportError:
    raise ImportError(
        "PyJWT package is required. Please install it using: pip install PyJWT"
    )

# Local application imports
import models
import schemas
import crud
import ai_utils
from models import (
    Base, User, UserProfile, EmotionAnalysis, EmotionHistory, 
    EmotionSession, RealtimeFrame, AudioAnalysis, MentalHealthScore, 
    TextAnalysis, TextAnalysisIntervention, MentalHealthAssessment, 
    MentalHealthIntervention, InterventionProgress
)
from database import SessionLocal, engine
from schemas import (
    UserCreate, UserUpdate, UserLogin, Token, TokenData, User, 
    UserProfileCreate, UserProfile, EmotionAnalysisCreate, EmotionAnalysis, 
    EmotionHistoryCreate, EmotionHistory, EmotionResponse, SessionType, 
    RealtimeFrameCreate, RealtimeFrame, EmotionSessionCreate, EmotionSession, 
    RealtimeAnalysisResponse, EmotionSummary, SessionSummaryResponse, 
    VideoAnalysisResult, AudioAnalysisResponse, EmotionTrendsResponse, 
    MentalHealthTrendsResponse, WellnessReportResponse, TextAnalysisRequest, 
    TextAnalysisResponse, MentalHealthAssessmentCreate, MentalHealthAssessmentResponse, 
    MentalHealthInterventionCreate, MentalHealthInterventionResponse, 
    InterventionProgressCreate, InterventionProgressResponse, StressTrackingResponse, 
    StressTrackingCreate, MeditationSessionResponse, MeditationSessionCreate, 
    MoodJournalResponse, MoodJournalCreate, CognitiveGameResponse, 
    CognitiveGameCreate, SleepRecordResponse, SleepRecordCreate, 
    TherapySessionResponse, TherapySessionCreate, EmergencyContactResponse, 
    EmergencyContactCreate, EmergencyAlertResponse, EmergencyAlertCreate
)
from security import (
    verify_password, get_password_hash, create_access_token,
    get_current_user, get_current_active_user
)
from emotion_utils import (
    save_uploaded_image,
    analyze_emotions,
    generate_mental_health_intervention,
    process_base64_image
)
from video_emotion_utils import (
    save_uploaded_video,
    extract_audio,
    analyze_facial_emotions,
    analyze_voice_emotions,
    transcribe_audio,
    analyze_text_sentiment,
    generate_comprehensive_analysis,
    generate_mental_health_intervention
)
from audio_analysis_utils import (
    save_audio_file,
    extract_audio_features,
    analyze_voice_emotion,
    transcribe_and_analyze_speech,
    assess_mental_state,
    generate_recommendations,
    generate_mental_health_scores
)
from text_analysis_utils import (
    analyze_text_sentiment,
    analyze_linguistic_patterns,
    extract_themes_and_concerns,
    generate_mental_health_assessment,
    generate_personalized_interventions
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT Configuration
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token expires in 30 minutes

# Load environment variables
load_dotenv()

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="NeuroCare API",
    description="""
    State-of-the-art mental health analysis and intervention system.
    
    ## Features
    * Real-time stress tracking and analysis
    * AI-powered meditation and breathing exercises
    * Comprehensive mood journaling with multi-modal analysis
    * Cognitive games for mental fitness
    * Advanced sleep coaching and analysis
    * AI therapy chatbot with personalized interventions
    * Emergency SOS system with contact management
    
    ## Authentication
    All endpoints require authentication using JWT tokens.
    """,
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# APIKeyHeader for Swagger UI
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication function
def authenticate_user(db: Session, email: str, password: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Get current user from token
async def get_current_user(token: str = Depends(api_key_header), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None or not token.startswith("Bearer "):
        raise credentials_exception
    token = token.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[os.getenv("ALGORITHM")])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/")
async def root():
    return {"status": "running", "message": "Mental Health Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user with this email exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login", response_model=schemas.Token)
def login(form_data: schemas.UserLogin, db: Session = Depends(get_db)):
    # Authenticate user
    user = authenticate_user(db, form_data.email, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/reset-password", response_model=schemas.PasswordResetResponse)
def request_password_reset(reset_data: schemas.PasswordReset, db: Session = Depends(get_db)):
    # Check if user exists
    user = db.query(models.User).filter(models.User.email == reset_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Generate OTP
    otp = generate_otp()
    
    # Set OTP expiration (10 minutes from now)
    expires_at = datetime.now() + timedelta(minutes=10)
    
    # Save OTP in database
    db_otp = models.PasswordResetOTP(
        user_id=user.id,
        otp=otp,
        expires_at=expires_at
    )
    db.add(db_otp)
    db.commit()
    
    # Send reset email with OTP
    email_sent = send_reset_email(user.email, otp)
    
    if not email_sent:
        # If email fails, delete the OTP and raise error
        db.delete(db_otp)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send reset email"
        )
    
    return {
        "message": "Password reset OTP sent to your email",
        "otp": otp  # Only in development, remove in production
    }

@app.post("/reset-password/verify", response_model=schemas.PasswordResetResponse)
def verify_password_reset(
    reset_data: schemas.PasswordResetVerify,
    db: Session = Depends(get_db)
):
    # Get user
    user = db.query(models.User).filter(models.User.email == reset_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Find the latest unused OTP for this user
    db_otp = db.query(models.PasswordResetOTP).filter(
        models.PasswordResetOTP.user_id == user.id,
        models.PasswordResetOTP.otp == reset_data.otp,
        models.PasswordResetOTP.is_used == False
    ).order_by(models.PasswordResetOTP.created_at.desc()).first()
    
    if not db_otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP"
        )
    
    # Check if OTP is expired
    if db_otp.is_expired:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired"
        )
    
    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    
    # Mark OTP as used
    db_otp.is_used = True
    
    db.commit()
    
    return {"message": "Password has been reset successfully"}

@app.post("/user/profile", response_model=schemas.UserProfile)
async def create_user_profile(
    profile: schemas.UserProfileCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if profile already exists
    db_profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()
    if db_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Profile already exists"
        )
    
    # Create new profile
    db_profile = models.UserProfile(
        user_id=current_user.id,
        **profile.dict()
    )
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.get("/user/profile", response_model=schemas.UserProfile)
async def get_user_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    return profile

@app.put("/user/profile", response_model=schemas.UserProfile)
async def update_user_profile(
    profile: schemas.UserProfileCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()
    if not db_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    for key, value in profile.dict().items():
        setattr(db_profile, key, value)
    
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.put("/user/me", response_model=schemas.User)
async def update_user(
    user_update: schemas.UserUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # If updating email, check if new email already exists
    if user_update.email and user_update.email != current_user.email:
        db_user = db.query(models.User).filter(models.User.email == user_update.email).first()
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        current_user.email = user_update.email

    # Update full name if provided
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name

    # Update password if provided
    if user_update.current_password and user_update.new_password:
        if not verify_password(user_update.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password"
            )
        current_user.hashed_password = get_password_hash(user_update.new_password)

    db.commit()
    db.refresh(current_user)
    return current_user

@app.get("/user/me", response_model=schemas.User)
async def get_user_info(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.get("/emotion/history", response_model=List[schemas.EmotionAnalysis])
async def get_emotion_history(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 10
):
    analyses = db.query(models.EmotionAnalysis).filter(
        models.EmotionAnalysis.user_id == current_user.id
    ).order_by(
        models.EmotionAnalysis.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return analyses

@app.get("/emotion/trends", response_model=Dict[str, List[schemas.EmotionHistory]])
async def get_emotion_trends(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    days: int = 7
):
    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Get emotion history within date range
    history = db.query(models.EmotionHistory).filter(
        models.EmotionHistory.user_id == current_user.id,
        models.EmotionHistory.created_at >= start_date,
        models.EmotionHistory.created_at <= end_date
    ).order_by(models.EmotionHistory.created_at.asc()).all()

    # Group by emotion type
    trends = {}
    for entry in history:
        if entry.emotion_type not in trends:
            trends[entry.emotion_type] = []
        trends[entry.emotion_type].append(entry)

    return trends

@app.get("/emotion/session/{session_id}/summary", response_model=schemas.SessionSummaryResponse)
def get_session_summary(session_id: int, db: Session = Depends(get_db)):
    session = crud.get_emotion_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    summary = json.loads(session.summary) if session.summary else {}
    interventions = session.interventions or []
    
    return schemas.SessionSummaryResponse(
        session=session,
        summary=schemas.EmotionSummary(**summary),
        interventions=interventions
    )

@app.get("/emotion/sessions", response_model=List[schemas.EmotionSession])
def get_user_sessions(
    skip: int = 0,
    limit: int = 10,
    session_type: Optional[schemas.SessionType] = None,
    db: Session = Depends(get_db)
):
    return crud.get_user_sessions(db, skip=skip, limit=limit, session_type=session_type)

@app.post("/emotion/analyze", response_model=schemas.EmotionResponse)
async def analyze_image_emotion(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Read and save the image
        image_contents = await file.read()
        image_path = save_uploaded_image(image_contents, current_user.id)

        # Analyze emotions
        emotions, dominant_emotion = analyze_emotions(image_path)
        
        # Generate intervention
        intervention = generate_intervention(emotions, dominant_emotion)

        # Create analysis record
        db_analysis = models.EmotionAnalysis(
            user_id=current_user.id,
            session_id=session_id,
            image_path=image_path,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            confidence_score=confidence_score,
            intervention=intervention,
            analysis_type="facial"  # Set analysis type to 'facial'
        )
        db.add(db_analysis)
        db.commit()  # Commit first so IDs are generated
        db.refresh(db_analysis)  # Refresh to get the ID

        history_records = []
        for emotion, score in emotions.items():
            history = models.EmotionHistory(
                user_id=current_user.id,
                emotion_analysis_id=db_analysis.id,  # Now analysis.id is set
                emotion_type=emotion,
                intensity=score
            )
            db.add(history)
            history_records.append(history)

        db.commit()  # Commit to save history records
        for h in history_records:
            db.refresh(h)  # Refresh to get IDs

        return {
            "analysis": db_analysis,
            "history": history_records,
            "intervention": intervention
        }

    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/emotion/analyze/base64", response_model=schemas.EmotionResponse)
async def analyze_base64_emotion(
    image_data: str = Body(..., embed=True),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Process base64 image
        image_bytes = process_base64_image(image_data)
        image_path = save_uploaded_image(image_bytes, current_user.id)

        # Analyze emotions
        emotions, dominant_emotion = analyze_emotions(image_path)
        
        # Generate intervention
        intervention = generate_intervention(emotions, dominant_emotion)

        # Create analysis record
        db_analysis = models.EmotionAnalysis(
            user_id=current_user.id,
            session_id=session_id,
            image_path=image_path,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            confidence_score=confidence_score,
            intervention=intervention,
            analysis_type="facial"  # Set analysis type to 'facial'
        )
        db.add(db_analysis)
        db.commit()  # Commit first so IDs are generated
        db.refresh(db_analysis)  # Refresh to get the ID

        history_records = []
        for emotion, score in emotions.items():
            history = models.EmotionHistory(
                user_id=current_user.id,
                emotion_analysis_id=db_analysis.id,  # Now analysis.id is set
                emotion_type=emotion,
                intensity=score
            )
            db.add(history)
            history_records.append(history)

        db.commit()  # Commit to save history records
        for h in history_records:
            db.refresh(h)  # Refresh to get IDs

        return {
            "analysis": db_analysis,
            "history": history_records,
            "intervention": intervention
        }

    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/emotion/analyze/video", response_model=schemas.VideoAnalysisResult)
async def analyze_video_emotion(
    video: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze emotions from a video recording including facial expressions, voice, and speech.
    The video should contain the user explaining their emotional state.
    """
    try:
        # Save uploaded video
        video_data = await video.read()
        video_path = save_uploaded_video(video_data, current_user.id)

        # Extract audio from video
        audio_path = extract_audio(video_path)

        # Analyze facial emotions from video frames
        facial_emotions = analyze_facial_emotions(video_path)

        # Analyze voice emotions from audio
        voice_emotions = analyze_voice_emotions(audio_path)

        # Transcribe speech and analyze sentiment
        transcription = transcribe_audio(audio_path)
        text_sentiment = analyze_text_sentiment(transcription)

        # Generate comprehensive analysis
        analysis = generate_comprehensive_analysis(
            facial_emotions=facial_emotions,
            voice_emotions=voice_emotions,
            text_sentiment=text_sentiment,
            transcription=transcription
        )

        # Generate mental health intervention using Mistral
        intervention = generate_mental_health_intervention(analysis)

        # Add intervention to analysis
        analysis["intervention"] = intervention

        # Save analysis to database (you'll need to create appropriate models)
        # ... (database storage code here)

        return analysis

    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        # Clean up temporary files
        try:
            if 'video_path' in locals():
                os.remove(video_path)
            if 'audio_path' in locals():
                os.remove(audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")

@app.get("/users/all", response_model=List[schemas.User], tags=["Users"])
def get_all_users(db: Session = Depends(get_db)):
    """
    Get all users - Open endpoint (no authentication required)
    """
    users = db.query(models.User).all()
    return [
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
        for user in users
    ]

@app.post("/mental-health/analyze/audio", response_model=schemas.AudioAnalysisResponse)
async def analyze_audio(
    audio: UploadFile = File(...),
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        # Save the uploaded audio file
        audio_path = await save_audio_file(audio)

        # Extract audio features
        features = await extract_audio_features(audio_path)

        # Analyze voice emotion
        emotion_analysis = await analyze_voice_emotion(audio_path)

        # Transcribe and analyze speech
        speech_analysis = await transcribe_and_analyze_speech(audio_path)

        # Assess mental state
        mental_state = await assess_mental_state(
            features,
            emotion_analysis,
            speech_analysis
        )

        # Generate mental health scores
        mental_health_scores = await generate_mental_health_scores(
            features,
            emotion_analysis,
            speech_analysis,
            mental_state
        )

        # Generate recommendations and follow-up questions
        recommendations, follow_up_questions = await generate_recommendations(
            emotion_analysis,
            mental_state,
            mental_health_scores
        )

        # Create session ID
        session_id = str(uuid.uuid4())

        # Create database record for audio analysis
        db_audio_analysis = models.AudioAnalysis(
            user_id=current_user.id,
            session_id=session_id,
            image_path=None,  # No image for audio analysis
            emotions=emotion_analysis,
            dominant_emotion=emotion_analysis["dominant_emotion"],
            confidence_score=emotion_analysis["confidence"],
            intervention=intervention,
            analysis_type="voice"  # Set analysis type to 'voice'
        )
        db.add(db_audio_analysis)

        # Create mental health scores record
        db_mental_health_scores = models.MentalHealthScore(
            audio_analysis_id=db_audio_analysis.id,
            **mental_health_scores
        )
        db.add(db_mental_health_scores)

        # Create recommendations record
        db_recommendations = models.AudioAnalysisRecommendation(
            audio_analysis_id=db_audio_analysis.id,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions,
            intervention_plan={
                "short_term": recommendations[:3],
                "long_term": recommendations[3:]
            }
        )
        db.add(db_recommendations)

        # Commit the transaction
        db.commit()
        db.refresh(db_audio_analysis)
        db.refresh(db_mental_health_scores)
        db.refresh(db_recommendations)

        # Create response
        response = schemas.AudioAnalysisResponse(
            session_id=session_id,
            timestamp=datetime.now(),
            audio_features=features,
            emotion_analysis=emotion_analysis,
            speech_content=speech_analysis,
            mental_state=mental_state,
            mental_health_scores=mental_health_scores,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions,
            intervention_plan=schemas.InterventionPlan(
                short_term=recommendations[:3],
                long_term=recommendations[3:]
            )
        )

        # Clean up the audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return response

    except Exception as e:
        # Rollback transaction in case of error
        db.rollback()
        # Clean up in case of error
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/mental-health/trends/emotions/{days}", response_model=schemas.EmotionTrendsResponse)
async def get_emotion_trends(
    days: int = Path(..., gt=0, le=365, description="Number of days to analyze"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get emotion trends analysis for the specified number of days.
    Limited to maximum 365 days of history.
    """
    try:
        trends = current_user.get_emotion_trends(days)
        if not trends:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No emotion analysis data found for the specified period"
            )
        return trends
    except Exception as e:
        logger.error(f"Error getting emotion trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/mental-health/trends/mental-health/{days}", response_model=schemas.MentalHealthTrendsResponse)
async def get_mental_health_trends(
    days: int = Path(..., gt=0, le=365, description="Number of days to analyze"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get mental health trends analysis from audio analyses for the specified number of days.
    Limited to maximum 365 days of history.
    """
    try:
        trends = current_user.get_mental_health_trends(days)
        if not trends:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No mental health analysis data found for the specified period"
            )
        return trends
    except Exception as e:
        logger.error(f"Error getting mental health trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/mental-health/trends/wellness-report/{days}", response_model=schemas.WellnessReportResponse)
async def get_wellness_report(
    days: int = Path(..., gt=0, le=365, description="Number of days to analyze"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a comprehensive wellness report combining emotion and mental health trends.
    Limited to maximum 365 days of history.
    """
    try:
        report = current_user.get_combined_wellness_report(days)
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analysis data found for the specified period"
            )
        return report
    except Exception as e:
        logger.error(f"Error generating wellness report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    return crud.create_user(db=db, user=user)

@app.get("/users/profile", response_model=schemas.UserProfile)
async def get_user_profile(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's profile."""
    profile = crud.get_user_profile(db, user_id=current_user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@app.post("/users/profile", response_model=schemas.UserProfile)
async def create_user_profile(
    profile: schemas.UserProfileCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create or update current user's profile."""
    return crud.create_user_profile(db=db, profile=profile, user_id=current_user.id)

@app.get("/mental-health/trends/emotions/{days}", response_model=schemas.EmotionTrendsResponse)
async def get_emotion_trends(
    days: int = Path(..., gt=0, le=365),
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get emotion trends analysis."""
    # Your existing implementation...

@app.get("/mental-health/trends/mental-health/{days}", response_model=schemas.MentalHealthTrendsResponse)
async def get_mental_health_trends(
    days: int = Path(..., gt=0, le=365),
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get mental health trends analysis."""
    # Your existing implementation...

@app.get("/mental-health/trends/wellness-report/{days}", response_model=schemas.WellnessReportResponse)
async def get_wellness_report(
    days: int = Path(..., gt=0, le=365),
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive wellness report."""
    # Your existing implementation...

@app.post("/mental-health/analyze/text", response_model=schemas.TextAnalysisResponse)
async def analyze_text(
    text_data: schemas.TextAnalysisRequest,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Analyze text input (journal entry, thoughts, feelings) for mental health insights.
    Provides comprehensive analysis including sentiment, linguistic patterns, themes,
    and personalized recommendations.
    """
    try:
        # Perform text analysis
        sentiment_analysis = analyze_text_sentiment(text_data.content)
        linguistic_analysis = analyze_linguistic_patterns(text_data.content)
        themes_analysis = extract_themes_and_concerns(text_data.content)
        
        # Generate comprehensive assessment
        assessment = await generate_mental_health_assessment(
            sentiment_analysis,
            linguistic_analysis,
            themes_analysis
        )
        
        # Generate personalized interventions
        interventions = await generate_personalized_interventions(assessment)
        
        # Create database record
        db_text_analysis = models.TextAnalysis(
            user_id=current_user.id,
            content=text_data.content,
            sentiment_score=sentiment_analysis["sentiment"]["polarity"],
            emotion_scores=sentiment_analysis["emotions"],
            linguistic_metrics=linguistic_analysis,
            identified_themes=themes_analysis["main_themes"],
            concerns=themes_analysis["potential_concerns"],
            risk_level=assessment["risk_level"],
            timestamp=datetime.now(),
            analysis_type="text"  # Set analysis type to 'text'
        )
        db.add(db_text_analysis)
        
        # Create intervention record
        db_intervention = models.TextAnalysisIntervention(
            text_analysis_id=db_text_analysis.id,
            recommendations=interventions["daily_practices"],
            goals=interventions["weekly_goals"],
            crisis_plan=interventions["crisis_plan"],
            reflection_prompts=interventions["reflection_prompts"],
            progress_metrics=interventions["progress_metrics"]
        )
        db.add(db_intervention)
        
        # Commit the transaction
        db.commit()
        db.refresh(db_text_analysis)
        db.refresh(db_intervention)
        
        # Create response
        response = schemas.TextAnalysisResponse(
            analysis_id=db_text_analysis.id,
            timestamp=db_text_analysis.timestamp,
            sentiment_analysis=sentiment_analysis,
            linguistic_analysis=linguistic_analysis,
            themes_analysis=themes_analysis,
            mental_health_assessment=assessment,
            personalized_interventions=interventions
        )
        
        return response
        
    except Exception as e:
        # Rollback transaction in case of error
        db.rollback()
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def generate_intervention(emotions, dominant_emotion):
    # Simple example: return a message based on dominant emotion
    return f"Suggested intervention for {dominant_emotion} based on detected emotions." 

@app.post("/mental-health/assess", response_model=schemas.MentalHealthAssessmentResponse)
async def create_mental_health_assessment(
    assessment: schemas.MentalHealthAssessmentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Create a new mental health assessment."""
    try:
        # Create the assessment
        db_assessment = crud.create_mental_health_assessment(
            db=db,
            user_id=current_user.id,
            assessment_data=assessment.dict()
        )
        
        # Generate AI insights
        current_metrics = {
            "depression": db_assessment.depression_score,
            "anxiety": db_assessment.anxiety_score,
            "stress": db_assessment.stress_score,
            "sleep_quality": db_assessment.sleep_quality_score,
            "emotional_regulation": db_assessment.emotional_regulation,
            "social_connection": db_assessment.social_connection,
            "resilience": db_assessment.resilience_score,
            "mindfulness": db_assessment.mindfulness_score
        }
        
        # Predict outcomes
        intervention_plan = {
            "type": "comprehensive",
            "duration": "ongoing",
            "focus_areas": list(current_metrics.keys())
        }
        
        predictions = ai_utils.predict_outcomes(current_metrics, intervention_plan)
        
        # Update assessment with AI insights
        db_assessment.ai_insights = {
            "predictions": predictions,
            "risk_level": "high" if any(rf["level"] == "high" for rf in predictions["risk_factors"]) else "medium",
            "recommendations": ai_utils.generate_recommendations({
                "areas_of_concern": [
                    {"metric": k, "deterioration": v}
                    for k, v in current_metrics.items()
                    if v > 0.7
                ]
            })
        }
        
        db.commit()
        db.refresh(db_assessment)
        
        return db_assessment
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating mental health assessment: {str(e)}"
        )

@app.get("/mental-health/assessments/history", response_model=List[schemas.MentalHealthAssessmentResponse])
async def get_assessment_history(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get user's mental health assessment history."""
    assessments = crud.get_mental_health_assessments(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    return assessments

@app.get("/mental-health/assessments/trends", response_model=schemas.MentalHealthTrendsResponse)
async def get_assessment_trends(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get trends in mental health assessments over time."""
    trends = crud.calculate_assessment_trends(
        db=db,
        user_id=current_user.id,
        days=days
    )
    
    if not trends:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No assessment data available for the specified period"
        )
    
    # Analyze trends and generate insights
    insights = ai_utils.analyze_mental_health_trends(trends)
    
    return {
        "trends": trends,
        "insights": insights,
        "risk_level": "high" if any(concern["deterioration"] > 0.2 for concern in insights["areas_of_concern"]) else "medium",
        "recommendations": insights["recommendations"]
    }

@app.post("/mental-health/interventions/track", response_model=schemas.InterventionProgressResponse)
async def track_intervention_progress(
    progress: schemas.InterventionProgressCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Track progress of a mental health intervention."""
    try:
        # Get the intervention
        intervention = crud.get_mental_health_intervention(
            db=db,
            intervention_id=progress.intervention_id
        )
        
        if not intervention:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Intervention not found"
            )
        
        # Create progress update
        db_progress = crud.create_intervention_progress(
            db=db,
            intervention_id=progress.intervention_id,
            progress_data=progress.dict()
        )
        
        # Generate AI feedback
        feedback = ai_utils.generate_intervention_feedback(
            progress_metrics=progress.dict(),
            intervention_type=intervention.intervention_type
        )
        
        return {
            "intervention": intervention,
            "ai_feedback": feedback,
            "next_steps": feedback["next_steps"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error tracking intervention progress: {str(e)}"
        )

# Stress Tracking Endpoints
@app.post("/stress/track", response_model=schemas.StressTrackingResponse, tags=["Stress Management"])
async def track_stress(
    stress_data: schemas.StressTrackingCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Track user's stress level with multi-modal analysis.
    
    This endpoint records stress levels using various analysis methods:
    - Facial expression analysis
    - Voice stress analysis
    - Text sentiment analysis
    
    Args:
        stress_data (schemas.StressTrackingCreate): Stress tracking data including:
            - stress_level: Float (0-1 scale)
            - source: String (facial/voice/text)
            - context: String (activity context)
            - location: Optional[String]
            - analysis results from different modalities
            
    Returns:
        schemas.StressTrackingResponse: Recorded stress tracking data
        
    Example:
        ```json
        {
            "stress_level": 0.75,
            "source": "facial",
            "context": "During work meeting",
            "location": "Office",
            "facial_analysis": {
                "tension_score": 0.8,
                "facial_markers": {...}
            }
        }
        ```
    """
    return crud.create_stress_tracking(db=db, user_id=current_user.id, stress_data=stress_data.dict())

@app.get("/stress/history", response_model=List[schemas.StressTrackingResponse], tags=["Stress Management"])
async def get_stress_history(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's stress tracking history with pagination.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.StressTrackingResponse]: List of stress tracking records
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "timestamp": "2024-03-15T10:30:00Z",
                "stress_level": 0.75,
                "source": "facial",
                "context": "During work meeting",
                "location": "Office",
                "facial_analysis": {...}
            }
        ]
        ```
    """
    return crud.get_stress_tracking(db=db, user_id=current_user.id, skip=skip, limit=limit)

# Meditation Session Endpoints
@app.post("/meditation/session", response_model=schemas.MeditationSessionResponse, tags=["Meditation"])
async def create_meditation(
    session_data: schemas.MeditationSessionCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new meditation session with AI-generated guidance.
    
    This endpoint creates a meditation session with:
    - AI-generated meditation script
    - Session type selection
    - Duration tracking
    - Progress monitoring
    
    Args:
        session_data (schemas.MeditationSessionCreate): Meditation session data including:
            - session_type: String (breathing/mindfulness/guided)
            - duration: Integer (minutes)
            - script: String (AI-generated meditation script)
            - audio_path: Optional[String]
            - completion_status: Optional[Float]
            - user_feedback: Optional[Dict]
            
    Returns:
        schemas.MeditationSessionResponse: Created meditation session
        
    Example:
        ```json
        {
            "session_type": "mindfulness",
            "duration": 15,
            "script": "Begin by finding a comfortable position...",
            "completion_status": 1.0,
            "user_feedback": {
                "effectiveness": 0.8,
                "difficulty": 0.3
            }
        }
        ```
    """
    return crud.create_meditation_session(db=db, user_id=current_user.id, session_data=session_data.dict())

@app.get("/meditation/history", response_model=List[schemas.MeditationSessionResponse], tags=["Meditation"])
async def get_meditation_history(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's meditation session history.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.MeditationSessionResponse]: List of meditation sessions
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "timestamp": "2024-03-15T10:30:00Z",
                "session_type": "mindfulness",
                "duration": 15,
                "script": "...",
                "completion_status": 1.0,
                "effectiveness_score": 0.8
            }
        ]
        ```
    """
    return crud.get_meditation_sessions(db=db, user_id=current_user.id, skip=skip, limit=limit)

# Mood Journal Endpoints
@app.post("/mood/journal", response_model=schemas.MoodJournalResponse, tags=["Mood Tracking"])
async def create_mood_journal(
    journal_data: schemas.MoodJournalCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new mood journal entry with multi-modal analysis.
    
    This endpoint creates a mood journal entry with:
    - Text content analysis
    - Audio emotion analysis
    - Facial expression analysis
    - Sentiment analysis
    - Theme identification
    
    Args:
        journal_data (schemas.MoodJournalCreate): Mood journal data including:
            - text_content: Optional[String]
            - audio_path: Optional[String]
            - facial_emotions: Optional[Dict]
            - mood_score: Float
            - dominant_emotions: List[String]
            - sentiment_analysis: Dict
            - themes: List[String]
            
    Returns:
        schemas.MoodJournalResponse: Created mood journal entry
        
    Example:
        ```json
        {
            "text_content": "Feeling productive today...",
            "mood_score": 0.8,
            "dominant_emotions": ["happy", "energetic"],
            "sentiment_analysis": {
                "positive": 0.8,
                "negative": 0.1
            },
            "themes": ["productivity", "motivation"]
        }
        ```
    """
    return crud.create_mood_journal(db=db, user_id=current_user.id, journal_data=journal_data.dict())

@app.get("/mood/history", response_model=List[schemas.MoodJournalResponse], tags=["Mood Tracking"])
async def get_mood_history(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's mood journal history.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.MoodJournalResponse]: List of mood journal entries
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "timestamp": "2024-03-15T10:30:00Z",
                "text_content": "Feeling productive today...",
                "mood_score": 0.8,
                "dominant_emotions": ["happy", "energetic"],
                "sentiment_analysis": {...},
                "themes": ["productivity", "motivation"]
            }
        ]
        ```
    """
    return crud.get_mood_journals(db=db, user_id=current_user.id, skip=skip, limit=limit)

# Cognitive Game Endpoints
@app.post("/cognitive/game", response_model=schemas.CognitiveGameResponse, tags=["Cognitive Training"])
async def create_cognitive_game(
    game_data: schemas.CognitiveGameCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Record a cognitive game session with performance metrics.
    
    This endpoint records cognitive game sessions with:
    - Game type and difficulty
    - Performance metrics
    - Cognitive assessment scores
    
    Args:
        game_data (schemas.CognitiveGameCreate): Game session data including:
            - game_type: String (memory/focus/puzzle)
            - difficulty_level: Integer
            - duration: Integer (seconds)
            - score: Float
            - accuracy: Float
            - reaction_time: Float
            - completion_status: Boolean
            - cognitive metrics
            
    Returns:
        schemas.CognitiveGameResponse: Recorded game session
        
    Example:
        ```json
        {
            "game_type": "memory",
            "difficulty_level": 3,
            "duration": 300,
            "score": 0.85,
            "accuracy": 0.9,
            "reaction_time": 0.5,
            "completion_status": true,
            "attention_score": 0.8,
            "memory_score": 0.85,
            "problem_solving_score": 0.75
        }
        ```
    """
    return crud.create_cognitive_game(db=db, user_id=current_user.id, game_data=game_data.dict())

@app.get("/cognitive/history", response_model=List[schemas.CognitiveGameResponse], tags=["Cognitive Training"])
async def get_cognitive_history(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's cognitive game history.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.CognitiveGameResponse]: List of cognitive game sessions
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "timestamp": "2024-03-15T10:30:00Z",
                "game_type": "memory",
                "difficulty_level": 3,
                "duration": 300,
                "score": 0.85,
                "accuracy": 0.9,
                "reaction_time": 0.5,
                "completion_status": true,
                "attention_score": 0.8,
                "memory_score": 0.85,
                "problem_solving_score": 0.75
            }
        ]
        ```
    """
    return crud.get_cognitive_games(db=db, user_id=current_user.id, skip=skip, limit=limit)

# Sleep Record Endpoints
@app.post("/sleep/record", response_model=schemas.SleepRecordResponse, tags=["Sleep Analysis"])
async def create_sleep_record(
    sleep_data: schemas.SleepRecordCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new sleep record with comprehensive analysis.
    
    This endpoint records sleep data with:
    - Sleep duration and quality metrics
    - Sleep environment data
    - Sleep habits tracking
    - AI-generated analysis and recommendations
    
    Args:
        sleep_data (schemas.SleepRecordCreate): Sleep record data including:
            - date: Date
            - sleep_duration: Float (hours)
            - sleep_quality: Float (0-1)
            - deep_sleep_duration: Float
            - rem_sleep_duration: Float
            - environment data
            - sleep habits
            - analysis and recommendations
            
    Returns:
        schemas.SleepRecordResponse: Created sleep record
        
    Example:
        ```json
        {
            "date": "2024-03-15",
            "sleep_duration": 7.5,
            "sleep_quality": 0.8,
            "deep_sleep_duration": 2.0,
            "rem_sleep_duration": 1.5,
            "room_temperature": 22.5,
            "noise_level": 0.2,
            "light_level": 0.1,
            "bedtime_routine": {
                "activities": ["reading", "meditation"],
                "duration": 30
            },
            "wake_up_time": "2024-03-15T07:00:00Z",
            "sleep_onset_time": "2024-03-14T23:30:00Z",
            "sleep_analysis": {
                "sleep_efficiency": 0.9,
                "sleep_cycles": 5
            },
            "recommendations": [
                "Maintain consistent sleep schedule",
                "Reduce screen time before bed"
            ]
        }
        ```
    """
    return crud.create_sleep_record(db=db, user_id=current_user.id, sleep_data=sleep_data.dict())

@app.get("/sleep/history", response_model=List[schemas.SleepRecordResponse], tags=["Sleep Analysis"])
async def get_sleep_history(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's sleep record history.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.SleepRecordResponse]: List of sleep records
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "date": "2024-03-15",
                "sleep_duration": 7.5,
                "sleep_quality": 0.8,
                "deep_sleep_duration": 2.0,
                "rem_sleep_duration": 1.5,
                "room_temperature": 22.5,
                "noise_level": 0.2,
                "light_level": 0.1,
                "bedtime_routine": {...},
                "wake_up_time": "2024-03-15T07:00:00Z",
                "sleep_onset_time": "2024-03-14T23:30:00Z",
                "sleep_analysis": {...},
                "recommendations": [...]
            }
        ]
        ```
    """
    return crud.get_sleep_records(db=db, user_id=current_user.id, skip=skip, limit=limit)

# Therapy Session Endpoints
@app.post("/therapy/session", response_model=schemas.TherapySessionResponse, tags=["Therapy"])
async def create_therapy_session(
    session_data: schemas.TherapySessionCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new therapy session with AI analysis.
    
    This endpoint creates a therapy session with:
    - Session type and duration
    - Chat content and analysis
    - Key concerns identification
    - Action items and follow-up planning
    
    Args:
        session_data (schemas.TherapySessionCreate): Therapy session data including:
            - session_type: String (chatbot/human)
            - duration: Integer (minutes)
            - topic: String
            - messages: List[Dict]
            - sentiment_analysis: Dict
            - key_concerns: List[String]
            - session_summary: String
            - action_items: List[String]
            - follow_up_needed: Boolean
            - escalation_level: Integer
            
    Returns:
        schemas.TherapySessionResponse: Created therapy session
        
    Example:
        ```json
        {
            "session_type": "chatbot",
            "duration": 30,
            "topic": "Stress Management",
            "messages": [
                {
                    "role": "user",
                    "content": "I've been feeling overwhelmed lately...",
                    "timestamp": "2024-03-15T10:00:00Z"
                },
                {
                    "role": "assistant",
                    "content": "I understand that feeling...",
                    "timestamp": "2024-03-15T10:00:05Z"
                }
            ],
            "sentiment_analysis": {
                "overall": "concerned",
                "emotions": ["anxiety", "stress"]
            },
            "key_concerns": [
                "Work pressure",
                "Sleep issues"
            ],
            "session_summary": "User discussed work-related stress...",
            "action_items": [
                "Practice daily meditation",
                "Set work boundaries"
            ],
            "follow_up_needed": true,
            "escalation_level": 2
        }
        ```
    """
    return crud.create_therapy_session(db=db, user_id=current_user.id, session_data=session_data.dict())

@app.get("/therapy/history", response_model=List[schemas.TherapySessionResponse], tags=["Therapy"])
async def get_therapy_history(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's therapy session history.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.TherapySessionResponse]: List of therapy sessions
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "timestamp": "2024-03-15T10:00:00Z",
                "session_type": "chatbot",
                "duration": 30,
                "topic": "Stress Management",
                "messages": [...],
                "sentiment_analysis": {...},
                "key_concerns": [...],
                "session_summary": "...",
                "action_items": [...],
                "follow_up_needed": true,
                "escalation_level": 2
            }
        ]
        ```
    """
    return crud.get_therapy_sessions(db=db, user_id=current_user.id, skip=skip, limit=limit)

# Emergency Contact Endpoints
@app.post("/emergency/contact", response_model=schemas.EmergencyContactResponse, tags=["Emergency"])
async def create_emergency_contact(
    contact_data: schemas.EmergencyContactCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add a new emergency contact.
    
    This endpoint adds an emergency contact with:
    - Contact details
    - Relationship information
    - Notification preferences
    
    Args:
        contact_data (schemas.EmergencyContactCreate): Contact data including:
            - name: String
            - relationship: String
            - phone_number: String
            - email: Optional[String]
            - is_primary: Boolean
            - notification preferences
            
    Returns:
        schemas.EmergencyContactResponse: Created emergency contact
        
    Example:
        ```json
        {
            "name": "John Doe",
            "relationship": "Family",
            "phone_number": "+1234567890",
            "email": "john@example.com",
            "is_primary": true,
            "notify_on_high_stress": true,
            "notify_on_crisis": true,
            "notify_on_missed_medication": true
        }
        ```
    """
    return crud.create_emergency_contact(db=db, user_id=current_user.id, contact_data=contact_data.dict())

@app.get("/emergency/contacts", response_model=List[schemas.EmergencyContactResponse], tags=["Emergency"])
async def get_emergency_contacts(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's emergency contacts.
    
    Returns:
        List[schemas.EmergencyContactResponse]: List of emergency contacts
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "name": "John Doe",
                "relationship": "Family",
                "phone_number": "+1234567890",
                "email": "john@example.com",
                "is_primary": true,
                "notify_on_high_stress": true,
                "notify_on_crisis": true,
                "notify_on_missed_medication": true
            }
        ]
        ```
    """
    return crud.get_emergency_contacts(db=db, user_id=current_user.id)

@app.put("/emergency/contact/{contact_id}", response_model=schemas.EmergencyContactResponse, tags=["Emergency"])
async def update_emergency_contact(
    contact_id: int,
    contact_data: schemas.EmergencyContactCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing emergency contact.
    
    Args:
        contact_id (int): ID of the contact to update
        contact_data (schemas.EmergencyContactCreate): Updated contact data
        
    Returns:
        schemas.EmergencyContactResponse: Updated emergency contact
        
    Raises:
        HTTPException: If contact not found
    """
    updated_contact = crud.update_emergency_contact(db=db, contact_id=contact_id, contact_data=contact_data.dict())
    if not updated_contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return updated_contact

@app.delete("/emergency/contact/{contact_id}", tags=["Emergency"])
async def delete_emergency_contact(
    contact_id: int,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete an emergency contact.
    
    Args:
        contact_id (int): ID of the contact to delete
        
    Returns:
        dict: Success message
        
    Raises:
        HTTPException: If contact not found
    """
    success = crud.delete_emergency_contact(db=db, contact_id=contact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"message": "Contact deleted successfully"}

# Emergency Alert Endpoints
@app.post("/emergency/alert", response_model=schemas.EmergencyAlertResponse, tags=["Emergency"])
async def create_emergency_alert(
    alert_data: schemas.EmergencyAlertCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new emergency alert.
    
    This endpoint creates an emergency alert with:
    - Alert type and severity
    - Description
    - Response tracking
    
    Args:
        alert_data (schemas.EmergencyAlertCreate): Alert data including:
            - alert_type: String (high_stress/crisis/missed_medication)
            - severity: Integer (1-5)
            - description: String
            - response details
            
    Returns:
        schemas.EmergencyAlertResponse: Created emergency alert
        
    Example:
        ```json
        {
            "alert_type": "high_stress",
            "severity": 4,
            "description": "User reported severe anxiety symptoms",
            "responded_by": "AI_System",
            "response_time": "2024-03-15T10:30:00Z",
            "resolution": "Contacted emergency contact"
        }
        ```
    """
    return crud.create_emergency_alert(db=db, user_id=current_user.id, alert_data=alert_data.dict())

@app.get("/emergency/alerts", response_model=List[schemas.EmergencyAlertResponse], tags=["Emergency"])
async def get_emergency_alerts(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve user's emergency alert history.
    
    Args:
        skip (int): Number of records to skip
        limit (int): Maximum number of records to return
        
    Returns:
        List[schemas.EmergencyAlertResponse]: List of emergency alerts
        
    Example Response:
        ```json
        [
            {
                "id": 1,
                "timestamp": "2024-03-15T10:30:00Z",
                "alert_type": "high_stress",
                "severity": 4,
                "description": "User reported severe anxiety symptoms",
                "responded_by": "AI_System",
                "response_time": "2024-03-15T10:30:00Z",
                "resolution": "Contacted emergency contact"
            }
        ]
        ```
    """
    return crud.get_emergency_alerts(db=db, user_id=current_user.id, skip=skip, limit=limit)

@app.put("/emergency/alert/{alert_id}", response_model=schemas.EmergencyAlertResponse, tags=["Emergency"])
async def update_emergency_alert(
    alert_id: int,
    alert_data: schemas.EmergencyAlertCreate,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing emergency alert.
    
    Args:
        alert_id (int): ID of the alert to update
        alert_data (schemas.EmergencyAlertCreate): Updated alert data
        
    Returns:
        schemas.EmergencyAlertResponse: Updated emergency alert
        
    Raises:
        HTTPException: If alert not found
    """
    updated_alert = crud.update_emergency_alert(db=db, alert_id=alert_id, alert_data=alert_data.dict())
    if not updated_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return updated_alert

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="NeuroCare API",
        version="1.0.0",
        openapi_version="3.0.0",  # Explicitly specify OpenAPI version
        description="State-of-the-art mental health analysis and intervention system",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    ) 