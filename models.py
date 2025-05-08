from sqlalchemy import Boolean, Column, Integer, String, DateTime, Float, ARRAY, ForeignKey, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime, timedelta

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    audio_analyses = relationship("AudioAnalysis", back_populates="user", cascade="all, delete-orphan")
    emotion_analyses = relationship("EmotionAnalysis", back_populates="user", cascade="all, delete-orphan")
    emotion_sessions = relationship("EmotionSession", back_populates="user", cascade="all, delete-orphan")
    emotion_history = relationship("EmotionHistory", back_populates="user", cascade="all, delete-orphan")
    user_profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    text_analyses = relationship("TextAnalysis", back_populates="user")

    def get_emotion_trends(self, days: int = 30):
        """Calculate emotion trends over specified number of days."""
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all emotion analyses in date range
        analyses = [ea for ea in self.emotion_analyses 
                   if start_date <= ea.created_at <= end_date]
        
        if not analyses:
            return None
            
        # Calculate daily emotion frequencies
        daily_emotions = {}
        for analysis in analyses:
            date = analysis.created_at.date()
            if date not in daily_emotions:
                daily_emotions[date] = {}
            
            emotion = analysis.dominant_emotion
            daily_emotions[date][emotion] = daily_emotions[date].get(emotion, 0) + 1
        
        # Calculate emotion trends
        emotion_trends = {
            "daily_frequencies": daily_emotions,
            "dominant_emotions": {},
            "emotion_stability": 0,
            "mood_variability": 0,
            "positive_ratio": 0
        }
        
        # Calculate overall statistics
        total_analyses = len(analyses)
        emotion_counts = {}
        positive_emotions = ["happy", "calm", "surprised"]
        positive_count = 0
        
        for analysis in analyses:
            emotion = analysis.dominant_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            if emotion.lower() in positive_emotions:
                positive_count += 1
        
        # Calculate dominant emotions and their percentages
        for emotion, count in emotion_counts.items():
            emotion_trends["dominant_emotions"][emotion] = count / total_analyses
        
        # Calculate positive ratio
        emotion_trends["positive_ratio"] = positive_count / total_analyses
        
        # Calculate mood variability (how often emotions change)
        emotion_changes = 0
        prev_emotion = None
        for analysis in sorted(analyses, key=lambda x: x.created_at):
            if prev_emotion and analysis.dominant_emotion != prev_emotion:
                emotion_changes += 1
            prev_emotion = analysis.dominant_emotion
        
        emotion_trends["mood_variability"] = emotion_changes / (total_analyses - 1) if total_analyses > 1 else 0
        
        return emotion_trends

    def get_mental_health_trends(self, days: int = 30):
        """Calculate mental health trends from audio analyses over specified number of days."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all audio analyses in date range
        analyses = [aa for aa in self.audio_analyses 
                   if start_date <= aa.created_at <= end_date]
        
        if not analyses:
            return None
        
        # Initialize trends dictionary
        mental_health_trends = {
            "daily_metrics": {},
            "overall_trends": {
                "stress_level": [],
                "anxiety_level": [],
                "mood_state": [],
                "emotional_stability": [],
                "social_engagement": [],
                "cognitive_load": [],
                "resilience": []
            },
            "risk_factors": {},
            "improvement_areas": [],
            "stability_metrics": {}
        }
        
        # Process each analysis
        for analysis in sorted(analyses, key=lambda x: x.created_at):
            date = analysis.created_at.date()
            
            # Daily metrics
            if date not in mental_health_trends["daily_metrics"]:
                mental_health_trends["daily_metrics"][date] = {
                    "stress_level": [],
                    "anxiety_level": [],
                    "mood_state": [],
                    "emotional_stability": [],
                    "social_engagement_level": [],
                    "cognitive_load": [],
                    "resilience_score": []
                }
            
            # Add metrics to daily tracking
            daily_metrics = mental_health_trends["daily_metrics"][date]
            daily_metrics["stress_level"].append(analysis.stress_level)
            daily_metrics["anxiety_level"].append(analysis.anxiety_level)
            daily_metrics["mood_state"].append(analysis.mood_state)
            daily_metrics["emotional_stability"].append(analysis.emotional_stability)
            daily_metrics["social_engagement_level"].append(analysis.social_engagement_level)
            daily_metrics["cognitive_load"].append(analysis.cognitive_load)
            daily_metrics["resilience_score"].append(analysis.resilience_score)
            
            # Track risk factors
            if analysis.risk_factors:
                for risk in analysis.risk_factors:
                    mental_health_trends["risk_factors"][risk] = mental_health_trends["risk_factors"].get(risk, 0) + 1
        
        # Calculate averages and trends
        for date, metrics in mental_health_trends["daily_metrics"].items():
            for metric, values in metrics.items():
                if values:
                    metrics[metric] = sum(values) / len(values)
        
        # Identify improvement areas
        latest_analysis = analyses[-1]
        threshold = 0.4  # Threshold for identifying areas needing improvement
        
        if latest_analysis.stress_level > threshold:
            mental_health_trends["improvement_areas"].append("stress_management")
        if latest_analysis.anxiety_level > threshold:
            mental_health_trends["improvement_areas"].append("anxiety_reduction")
        if latest_analysis.emotional_stability < threshold:
            mental_health_trends["improvement_areas"].append("emotional_regulation")
        if latest_analysis.social_engagement_level < threshold:
            mental_health_trends["improvement_areas"].append("social_engagement")
        if latest_analysis.resilience_score < threshold:
            mental_health_trends["improvement_areas"].append("resilience_building")
        
        # Calculate stability metrics
        for metric in ["stress_level", "anxiety_level", "emotional_stability", "resilience_score"]:
            values = [getattr(a, metric) for a in analyses]
            if values:
                avg = sum(values) / len(values)
                variance = sum((x - avg) ** 2 for x in values) / len(values)
                mental_health_trends["stability_metrics"][metric] = {
                    "average": avg,
                    "variance": variance,
                    "trend": "improving" if values[-1] > values[0] else "declining"
                }
        
        return mental_health_trends

    def get_combined_wellness_report(self, days: int = 30):
        """Generate a comprehensive wellness report combining emotion and mental health trends."""
        emotion_trends = self.get_emotion_trends(days)
        mental_health_trends = self.get_mental_health_trends(days)
        
        if not emotion_trends or not mental_health_trends:
            return None
        
        wellness_report = {
            "overall_status": {
                "emotional_wellbeing": None,
                "mental_health": None,
                "social_engagement": None,
                "resilience": None
            },
            "trends": {
                "emotions": emotion_trends,
                "mental_health": mental_health_trends
            },
            "recommendations": [],
            "risk_level": "low"  # Can be low, moderate, high
        }
        
        # Calculate overall emotional wellbeing
        wellness_report["overall_status"]["emotional_wellbeing"] = {
            "score": emotion_trends["positive_ratio"],
            "stability": 1 - emotion_trends["mood_variability"]
        }
        
        # Calculate overall mental health
        latest_metrics = list(mental_health_trends["daily_metrics"].values())[-1]
        wellness_report["overall_status"]["mental_health"] = {
            "stress_level": latest_metrics["stress_level"],
            "anxiety_level": latest_metrics["anxiety_level"],
            "emotional_stability": latest_metrics["emotional_stability"]
        }
        
        # Set social engagement and resilience
        wellness_report["overall_status"]["social_engagement"] = latest_metrics["social_engagement_level"]
        wellness_report["overall_status"]["resilience"] = latest_metrics["resilience_score"]
        
        # Determine risk level
        risk_factors_count = len(mental_health_trends["risk_factors"])
        high_risk_metrics = sum(1 for metric in latest_metrics.values() 
                              if isinstance(metric, (int, float)) and metric > 0.7)
        
        if risk_factors_count > 3 or high_risk_metrics > 2:
            wellness_report["risk_level"] = "high"
        elif risk_factors_count > 1 or high_risk_metrics > 0:
            wellness_report["risk_level"] = "moderate"
        
        # Generate recommendations based on trends
        if wellness_report["overall_status"]["emotional_wellbeing"]["score"] < 0.4:
            wellness_report["recommendations"].append("Focus on activities that bring joy and positive emotions")
        
        if wellness_report["overall_status"]["mental_health"]["stress_level"] > 0.6:
            wellness_report["recommendations"].append("Implement stress management techniques")
        
        if wellness_report["overall_status"]["social_engagement"] < 0.5:
            wellness_report["recommendations"].append("Increase social interactions and support network engagement")
        
        if wellness_report["overall_status"]["resilience"] < 0.5:
            wellness_report["recommendations"].append("Work on building resilience through small daily challenges")
        
        return wellness_report

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)  # Male, Female, Non-binary, Prefer not to say
    sleep_hours_actual = Column(Float)  # Current sleep hours
    sleep_hours_target = Column(Float)  # Target sleep hours
    goals = Column(ARRAY(String))  # Array of selected goals
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship
    user = relationship("User", back_populates="user_profile")

class PasswordResetOTP(Base):
    __tablename__ = "password_reset_otps"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    otp = Column(String(6), index=True)  # 6-digit OTP
    expires_at = Column(DateTime(timezone=True))
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    @property
    def is_expired(self):
        return datetime.now().astimezone() > self.expires_at

class EmotionAnalysis(Base):
    __tablename__ = "emotion_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    session_id = Column(String, unique=True, index=True)
    image_path = Column(String)  # Path to stored image
    emotions = Column(JSON)  # Detailed emotion analysis
    dominant_emotion = Column(String)
    confidence_score = Column(Float)
    intervention = Column(String)  # AI-generated intervention
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="emotion_analyses")
    emotion_history = relationship("EmotionHistory", back_populates="emotion_analysis", cascade="all, delete-orphan")

class EmotionHistory(Base):
    __tablename__ = "emotion_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    emotion_analysis_id = Column(Integer, ForeignKey("emotion_analyses.id"))
    emotion_type = Column(String)  # e.g., "happy", "sad", etc.
    intensity = Column(Float)  # Emotion intensity (0-1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="emotion_history")
    emotion_analysis = relationship("EmotionAnalysis", back_populates="emotion_history")

class EmotionSession(Base):
    __tablename__ = "emotion_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    session_id = Column(String, unique=True, index=True)
    session_type = Column(String)  # 'realtime' or 'upload'
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    average_emotions = Column(JSON)  # Average emotion scores for the session
    emotion_timeline = Column(JSON)  # Timeline of emotions during session
    summary = Column(String)  # Session summary
    interventions = Column(ARRAY(String))  # List of interventions provided
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="emotion_sessions")
    realtime_frames = relationship("RealtimeFrame", back_populates="session", cascade="all, delete-orphan")

class RealtimeFrame(Base):
    __tablename__ = "realtime_frames"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("emotion_sessions.id"), index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    emotions = Column(JSON)  # Emotion scores for this frame
    dominant_emotion = Column(String)
    confidence_score = Column(Float)  # Confidence in emotion detection
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship
    session = relationship("EmotionSession", back_populates="realtime_frames")

class AudioAnalysis(Base):
    __tablename__ = "audio_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Audio Features
    pitch_mean = Column(Float)
    pitch_std = Column(Float)
    energy = Column(Float)
    tempo = Column(Float)
    speech_rate = Column(Float)
    pause_ratio = Column(Float)
    voice_quality = Column(Float)
    
    # Emotion Analysis
    arousal = Column(Float)
    valence = Column(Float)
    dominant_emotion = Column(String)
    confidence_score = Column(Float)
    emotion_scores = Column(JSON)
    
    # Speech Content
    transcription = Column(Text)
    sentiment_score = Column(Float)
    key_phrases = Column(JSON)
    hesitation_count = Column(Integer)
    word_per_minute = Column(Float)
    
    # Mental Health Metrics
    stress_level = Column(Float)
    anxiety_level = Column(Float)
    depression_indicators = Column(JSON)
    mood_state = Column(String)
    energy_level = Column(Float)
    coherence_score = Column(Float)
    emotional_stability = Column(Float)
    sleep_quality_indicator = Column(Float)
    social_engagement_level = Column(Float)
    cognitive_load = Column(Float)
    resilience_score = Column(Float)
    
    # Analysis Results
    recommendations = Column(JSON)
    follow_up_questions = Column(JSON)
    risk_factors = Column(JSON)
    intervention_plan = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="audio_analyses")
    mental_health_scores = relationship("MentalHealthScore", back_populates="audio_analysis", cascade="all, delete-orphan")

class MentalHealthScore(Base):
    __tablename__ = "mental_health_scores"

    id = Column(Integer, primary_key=True, index=True)
    audio_analysis_id = Column(Integer, ForeignKey("audio_analyses.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Detailed Mental Health Metrics
    anxiety_score = Column(Float)
    depression_score = Column(Float)
    stress_score = Column(Float)
    emotional_regulation = Column(Float)
    social_connection = Column(Float)
    mindfulness = Column(Float)
    sleep_quality = Column(Float)
    cognitive_performance = Column(Float)
    resilience = Column(Float)
    life_satisfaction = Column(Float)
    
    # Relationships
    audio_analysis = relationship("AudioAnalysis", back_populates="mental_health_scores")

class TextAnalysis(Base):
    __tablename__ = "text_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    sentiment_score = Column(Float)
    emotion_scores = Column(JSON)
    linguistic_metrics = Column(JSON)
    identified_themes = Column(ARRAY(String))
    concerns = Column(JSON)
    risk_level = Column(Integer)
    timestamp = Column(DateTime, default=datetime.now)

    # Relationships
    user = relationship("User", back_populates="text_analyses")
    intervention = relationship("TextAnalysisIntervention", back_populates="text_analysis", uselist=False)

class TextAnalysisIntervention(Base):
    __tablename__ = "text_analysis_interventions"

    id = Column(Integer, primary_key=True, index=True)
    text_analysis_id = Column(Integer, ForeignKey("text_analyses.id"))
    recommendations = Column(ARRAY(String))
    goals = Column(ARRAY(String))
    crisis_plan = Column(JSON)
    reflection_prompts = Column(ARRAY(String))
    progress_metrics = Column(ARRAY(String))
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    text_analysis = relationship("TextAnalysis", back_populates="intervention") 