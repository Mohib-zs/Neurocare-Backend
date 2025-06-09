from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
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

def create_mental_health_assessment(db: Session, user_id: int, assessment_data: dict):
    """Create a new mental health assessment."""
    db_assessment = models.MentalHealthAssessment(
        user_id=user_id,
        **assessment_data
    )
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    return db_assessment

def get_mental_health_assessments(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100
):
    """Get mental health assessments for a user."""
    return db.query(models.MentalHealthAssessment)\
        .filter(models.MentalHealthAssessment.user_id == user_id)\
        .order_by(models.MentalHealthAssessment.timestamp.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_mental_health_assessment(db: Session, assessment_id: int):
    """Get a specific mental health assessment."""
    return db.query(models.MentalHealthAssessment)\
        .filter(models.MentalHealthAssessment.id == assessment_id)\
        .first()

def create_mental_health_intervention(
    db: Session,
    assessment_id: int,
    intervention_data: dict
):
    """Create a new mental health intervention."""
    db_intervention = models.MentalHealthIntervention(
        assessment_id=assessment_id,
        **intervention_data
    )
    db.add(db_intervention)
    db.commit()
    db.refresh(db_intervention)
    return db_intervention

def get_mental_health_interventions(
    db: Session,
    assessment_id: int,
    skip: int = 0,
    limit: int = 100
):
    """Get interventions for a specific assessment."""
    return db.query(models.MentalHealthIntervention)\
        .filter(models.MentalHealthIntervention.assessment_id == assessment_id)\
        .order_by(models.MentalHealthIntervention.timestamp.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def create_intervention_progress(
    db: Session,
    intervention_id: int,
    progress_data: dict
):
    """Create a new progress update for an intervention."""
    db_progress = models.InterventionProgress(
        intervention_id=intervention_id,
        **progress_data
    )
    db.add(db_progress)
    db.commit()
    db.refresh(db_progress)
    return db_progress

def get_intervention_progress(
    db: Session,
    intervention_id: int,
    skip: int = 0,
    limit: int = 100
):
    """Get progress updates for a specific intervention."""
    return db.query(models.InterventionProgress)\
        .filter(models.InterventionProgress.intervention_id == intervention_id)\
        .order_by(models.InterventionProgress.timestamp.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def calculate_assessment_trends(
    db: Session,
    user_id: int,
    days: int = 30
):
    """Calculate trends in mental health assessments over time."""
    start_date = datetime.utcnow() - timedelta(days=days)
    assessments = db.query(models.MentalHealthAssessment)\
        .filter(
            models.MentalHealthAssessment.user_id == user_id,
            models.MentalHealthAssessment.timestamp >= start_date
        )\
        .order_by(models.MentalHealthAssessment.timestamp.asc())\
        .all()
    
    if not assessments:
        return None
    
    trends = {
        "depression": [],
        "anxiety": [],
        "stress": [],
        "sleep_quality": [],
        "emotional_regulation": [],
        "social_connection": [],
        "resilience": [],
        "mindfulness": []
    }
    
    for assessment in assessments:
        trends["depression"].append(assessment.depression_score)
        trends["anxiety"].append(assessment.anxiety_score)
        trends["stress"].append(assessment.stress_score)
        trends["sleep_quality"].append(assessment.sleep_quality_score)
        trends["emotional_regulation"].append(assessment.emotional_regulation)
        trends["social_connection"].append(assessment.social_connection)
        trends["resilience"].append(assessment.resilience_score)
        trends["mindfulness"].append(assessment.mindfulness_score)
    
    return trends

# Stress Tracking CRUD
def create_stress_tracking(db: Session, user_id: int, stress_data: Dict[str, Any]) -> models.StressTracking:
    db_stress = models.StressTracking(
        user_id=user_id,
        **stress_data
    )
    db.add(db_stress)
    db.commit()
    db.refresh(db_stress)
    return db_stress

def get_stress_tracking(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.StressTracking]:
    return db.query(models.StressTracking).filter(
        models.StressTracking.user_id == user_id
    ).order_by(
        models.StressTracking.timestamp.desc()
    ).offset(skip).limit(limit).all()

# Meditation Session CRUD
def create_meditation_session(db: Session, user_id: int, session_data: Dict[str, Any]) -> models.MeditationSession:
    db_session = models.MeditationSession(
        user_id=user_id,
        **session_data
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_meditation_sessions(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.MeditationSession]:
    return db.query(models.MeditationSession).filter(
        models.MeditationSession.user_id == user_id
    ).order_by(
        models.MeditationSession.timestamp.desc()
    ).offset(skip).limit(limit).all()

# Mood Journal CRUD
def create_mood_journal(db: Session, user_id: int, journal_data: Dict[str, Any]) -> models.MoodJournal:
    db_journal = models.MoodJournal(
        user_id=user_id,
        **journal_data
    )
    db.add(db_journal)
    db.commit()
    db.refresh(db_journal)
    return db_journal

def get_mood_journals(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.MoodJournal]:
    return db.query(models.MoodJournal).filter(
        models.MoodJournal.user_id == user_id
    ).order_by(
        models.MoodJournal.timestamp.desc()
    ).offset(skip).limit(limit).all()

# Cognitive Game CRUD
def create_cognitive_game(db: Session, user_id: int, game_data: Dict[str, Any]) -> models.CognitiveGame:
    db_game = models.CognitiveGame(
        user_id=user_id,
        **game_data
    )
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return db_game

def get_cognitive_games(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.CognitiveGame]:
    return db.query(models.CognitiveGame).filter(
        models.CognitiveGame.user_id == user_id
    ).order_by(
        models.CognitiveGame.timestamp.desc()
    ).offset(skip).limit(limit).all()

# Sleep Record CRUD
def create_sleep_record(db: Session, user_id: int, sleep_data: Dict[str, Any]) -> models.SleepRecord:
    db_sleep = models.SleepRecord(
        user_id=user_id,
        **sleep_data
    )
    db.add(db_sleep)
    db.commit()
    db.refresh(db_sleep)
    return db_sleep

def get_sleep_records(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.SleepRecord]:
    return db.query(models.SleepRecord).filter(
        models.SleepRecord.user_id == user_id
    ).order_by(
        models.SleepRecord.date.desc()
    ).offset(skip).limit(limit).all()

# Therapy Session CRUD
def create_therapy_session(db: Session, user_id: int, session_data: Dict[str, Any]) -> models.TherapySession:
    db_session = models.TherapySession(
        user_id=user_id,
        **session_data
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_therapy_sessions(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.TherapySession]:
    return db.query(models.TherapySession).filter(
        models.TherapySession.user_id == user_id
    ).order_by(
        models.TherapySession.timestamp.desc()
    ).offset(skip).limit(limit).all()

# Emergency Contact CRUD
def create_emergency_contact(db: Session, user_id: int, contact_data: Dict[str, Any]) -> models.EmergencyContact:
    db_contact = models.EmergencyContact(
        user_id=user_id,
        **contact_data
    )
    db.add(db_contact)
    db.commit()
    db.refresh(db_contact)
    return db_contact

def get_emergency_contacts(db: Session, user_id: int) -> List[models.EmergencyContact]:
    return db.query(models.EmergencyContact).filter(
        models.EmergencyContact.user_id == user_id
    ).all()

def update_emergency_contact(db: Session, contact_id: int, contact_data: Dict[str, Any]) -> models.EmergencyContact:
    db_contact = db.query(models.EmergencyContact).filter(
        models.EmergencyContact.id == contact_id
    ).first()
    if db_contact:
        for key, value in contact_data.items():
            setattr(db_contact, key, value)
        db.commit()
        db.refresh(db_contact)
    return db_contact

def delete_emergency_contact(db: Session, contact_id: int) -> bool:
    db_contact = db.query(models.EmergencyContact).filter(
        models.EmergencyContact.id == contact_id
    ).first()
    if db_contact:
        db.delete(db_contact)
        db.commit()
        return True
    return False

# Emergency Alert CRUD
def create_emergency_alert(db: Session, user_id: int, alert_data: Dict[str, Any]) -> models.EmergencyAlert:
    db_alert = models.EmergencyAlert(
        user_id=user_id,
        **alert_data
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_emergency_alerts(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.EmergencyAlert]:
    return db.query(models.EmergencyAlert).filter(
        models.EmergencyAlert.user_id == user_id
    ).order_by(
        models.EmergencyAlert.timestamp.desc()
    ).offset(skip).limit(limit).all()

def update_emergency_alert(db: Session, alert_id: int, alert_data: Dict[str, Any]) -> models.EmergencyAlert:
    db_alert = db.query(models.EmergencyAlert).filter(
        models.EmergencyAlert.id == alert_id
    ).first()
    if db_alert:
        for key, value in alert_data.items():
            setattr(db_alert, key, value)
        db.commit()
        db.refresh(db_alert)
    return db_alert 