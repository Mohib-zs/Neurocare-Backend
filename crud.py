from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import models, schemas
from security import get_password_hash

def create_emotion_session(
    db: Session,
    user_id: int,
    session_type: schemas.SessionType
) -> models.EmotionSession:
    db_session = models.EmotionSession(
        user_id=user_id,
        session_type=session_type,
        start_time=datetime.now()
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_emotion_session(db: Session, session_id: int) -> Optional[models.EmotionSession]:
    return db.query(models.EmotionSession).filter(models.EmotionSession.id == session_id).first()

def get_user_sessions(
    db: Session,
    skip: int = 0,
    limit: int = 10,
    session_type: Optional[schemas.SessionType] = None
) -> List[models.EmotionSession]:
    query = db.query(models.EmotionSession)
    if session_type:
        query = query.filter(models.EmotionSession.session_type == session_type)
    return query.offset(skip).limit(limit).all()

def create_realtime_frame(
    db: Session,
    session_id: int,
    emotions: Dict[str, float],
    dominant_emotion: str,
    confidence_score: float
) -> models.RealtimeFrame:
    db_frame = models.RealtimeFrame(
        session_id=session_id,
        emotions=emotions,
        dominant_emotion=dominant_emotion,
        confidence_score=confidence_score,
        timestamp=datetime.now()
    )
    db.add(db_frame)
    db.commit()
    db.refresh(db_frame)
    return db_frame

def get_session_frames(
    db: Session,
    session_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[models.RealtimeFrame]:
    return db.query(models.RealtimeFrame)\
        .filter(models.RealtimeFrame.session_id == session_id)\
        .order_by(models.RealtimeFrame.timestamp.asc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def update_session_summary(
    db: Session,
    session_id: int,
    summary: Dict,
    interventions: List[str]
) -> models.EmotionSession:
    session = get_emotion_session(db, session_id)
    if session:
        session.summary = summary
        session.interventions = interventions
        session.end_time = datetime.now()
        db.commit()
        db.refresh(session)
    return session

def get_user_emotion_trends(
    db: Session,
    user_id: int,
    days: int = 7
) -> Dict:
    start_date = datetime.now() - timedelta(days=days)
    
    # Get all sessions in the date range
    sessions = db.query(models.EmotionSession)\
        .filter(
            models.EmotionSession.user_id == user_id,
            models.EmotionSession.start_time >= start_date
        )\
        .all()
    
    # Collect all frames from these sessions
    frames = []
    for session in sessions:
        session_frames = get_session_frames(db, session.id)
        frames.extend(session_frames)
    
    # Aggregate emotions by day
    daily_emotions = {}
    for frame in frames:
        day = frame.timestamp.date().isoformat()
        if day not in daily_emotions:
            daily_emotions[day] = {
                "count": 0,
                "emotions": {k: 0.0 for k in frame.emotions.keys()}
            }
        
        daily_emotions[day]["count"] += 1
        for emotion, value in frame.emotions.items():
            daily_emotions[day]["emotions"][emotion] += value
    
    # Calculate averages
    for day_data in daily_emotions.values():
        for emotion in day_data["emotions"]:
            day_data["emotions"][emotion] /= day_data["count"]
    
    return daily_emotions

def get_user(db: Session, user_id: int):
    """Get user by ID."""
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    """Get user by email."""
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    """Get list of users."""
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    """Create new user."""
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

def get_user_profile(db: Session, user_id: int):
    """Get user profile."""
    return db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()

def create_user_profile(db: Session, profile: schemas.UserProfileCreate, user_id: int):
    """Create or update user profile."""
    db_profile = get_user_profile(db, user_id)
    if db_profile:
        # Update existing profile
        for key, value in profile.dict().items():
            setattr(db_profile, key, value)
    else:
        # Create new profile
        db_profile = models.UserProfile(**profile.dict(), user_id=user_id)
        db.add(db_profile)
    
    db.commit()
    db.refresh(db_profile)
    return db_profile

def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate):
    """Update user information."""
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    # Update user fields
    if user_update.email:
        db_user.email = user_update.email
    if user_update.full_name:
        db_user.full_name = user_update.full_name
    if user_update.new_password:
        db_user.hashed_password = get_password_hash(user_update.new_password)
    
    db.commit()
    db.refresh(db_user)
    return db_user 