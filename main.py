from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Body, Path
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from database import SessionLocal, engine, get_db
import models
import schemas
from fastapi.middleware.cors import CORSMiddleware
from datetime import timedelta, datetime
from security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    get_current_active_user,
    authenticate_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
import os
from dotenv import load_dotenv
import secrets
from emotion_utils import (
    save_uploaded_image,
    analyze_emotions,
    generate_mental_health_intervention,
    process_base64_image
)
from fastapi.responses import JSONResponse
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
import logging
from sqlalchemy import func
from audio_analysis_utils import (
    save_audio_file,
    extract_audio_features,
    analyze_voice_emotion,
    transcribe_and_analyze_speech,
    assess_mental_state,
    generate_recommendations,
    generate_mental_health_scores
)
import uuid
import crud
from text_analysis_utils import (
    analyze_text_sentiment,
    analyze_linguistic_patterns,
    extract_themes_and_concerns,
    generate_mental_health_assessment,
    generate_personalized_interventions
)

load_dotenv()

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Mental Health Analysis API",
    description="API for analyzing mental health through facial expressions, voice, and video analysis",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Get current user from token
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[os.getenv("ALGORITHM")])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

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
        analysis = models.EmotionAnalysis(
            user_id=current_user.id,
            image_path=image_path,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            intervention=intervention
        )
        db.add(analysis)

        # Create emotion history records
        for emotion, score in emotions.items():
            history = models.EmotionHistory(
                user_id=current_user.id,
                emotion_analysis_id=analysis.id,
                emotion_type=emotion,
                intensity=score
            )
            db.add(history)

        db.commit()

        return {
            "analysis": analysis,
            "history": [history],
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
        analysis = models.EmotionAnalysis(
            user_id=current_user.id,
            image_path=image_path,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            intervention=intervention
        )
        db.add(analysis)

        # Create emotion history records
        for emotion, score in emotions.items():
            history = models.EmotionHistory(
                user_id=current_user.id,
                emotion_analysis_id=analysis.id,
                emotion_type=emotion,
                intensity=score
            )
            db.add(history)

        db.commit()

        return {
            "analysis": analysis,
            "history": [history],
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
            # Audio Features
            pitch_mean=features["pitch_mean"],
            pitch_std=features["pitch_std"],
            energy=features["energy"],
            tempo=features["tempo"],
            speech_rate=features["speech_rate"],
            pause_ratio=features["pause_ratio"],
            voice_quality=features["voice_quality"],
            # Emotion Analysis
            arousal=emotion_analysis["arousal"],
            valence=emotion_analysis["valence"],
            dominant_emotion=emotion_analysis["dominant_emotion"],
            confidence_score=emotion_analysis["confidence"],
            emotion_scores=emotion_analysis["emotion_scores"],
            # Speech Content
            transcription=speech_analysis["transcription"],
            sentiment_score=speech_analysis["sentiment_score"],
            key_phrases=speech_analysis["key_phrases"],
            hesitation_count=speech_analysis["hesitation_count"],
            word_per_minute=speech_analysis["word_per_minute"],
            # Mental Health Metrics
            stress_level=mental_state["stress_level"],
            anxiety_level=mental_state["anxiety_level"],
            depression_indicators=mental_state["depression_indicators"],
            mood_state=mental_state["mood_state"],
            energy_level=mental_state["energy_level"],
            coherence_score=mental_state["coherence_score"],
            emotional_stability=mental_state["emotional_stability"],
            sleep_quality_indicator=mental_state["sleep_quality_indicator"],
            social_engagement_level=mental_state["social_engagement_level"],
            cognitive_load=mental_state["cognitive_load"],
            resilience_score=mental_state["resilience_score"]
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
            timestamp=datetime.now()
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