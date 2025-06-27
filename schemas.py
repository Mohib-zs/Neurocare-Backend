from pydantic import BaseModel, EmailStr, constr, Field
from typing import Optional, List, Dict, Union, Any
from datetime import datetime, date
from enum import Enum

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    NON_BINARY = "Non-binary"
    PREFER_NOT_TO_SAY = "Prefer not to say"

class Goal(str, Enum):
    REDUCE_STRESS = "Reduce Stress & Anxiety"
    IMPROVE_SLEEP = "Improve Sleep Quality"
    ENHANCE_MOOD = "Enhance Mood"
    BOOST_FOCUS = "Boost Focus"
    BETTER_RELATIONSHIPS = "Better Relationships"
    PRACTICE_MINDFULNESS = "Practice Mindfulness"

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    full_name: str
    password: constr(min_length=6)

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[constr(min_length=6)] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetVerify(BaseModel):
    email: EmailStr
    otp: str
    new_password: constr(min_length=6)

class PasswordResetResponse(BaseModel):
    message: str
    otp: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class User(UserBase):
    id: int
    full_name: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserProfileBase(BaseModel):
    name: str
    age: int
    gender: Gender
    sleep_hours_actual: float
    sleep_hours_target: float
    goals: List[Goal]

class UserProfileCreate(UserProfileBase):
    pass

class UserProfile(UserProfileBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class EmotionData(BaseModel):
    emotion_type: str
    intensity: float

class EmotionAnalysisBase(BaseModel):
    emotions: Dict[str, float]
    dominant_emotion: str
    intervention: str

class EmotionAnalysisCreate(EmotionAnalysisBase):
    pass

class EmotionAnalysis(EmotionAnalysisBase):
    id: int
    user_id: int
    image_path: str
    created_at: datetime

    class Config:
        from_attributes = True

class EmotionHistoryBase(BaseModel):
    emotion_type: str
    intensity: float

class EmotionHistoryCreate(EmotionHistoryBase):
    emotion_analysis_id: int

class EmotionHistory(EmotionHistoryBase):
    id: int
    user_id: int
    emotion_analysis_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class EmotionResponse(BaseModel):
    analysis: EmotionAnalysis
    history: List[EmotionHistory]
    intervention: str

class SessionType(str, Enum):
    REALTIME = "realtime"
    UPLOAD = "upload"

class RealtimeFrameBase(BaseModel):
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence_score: float

class RealtimeFrameCreate(RealtimeFrameBase):
    pass

class RealtimeFrame(RealtimeFrameBase):
    id: int
    session_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class EmotionSessionBase(BaseModel):
    session_type: SessionType
    average_emotions: Optional[Dict[str, float]] = None
    emotion_timeline: Optional[Dict[str, List[float]]] = None
    summary: Optional[str] = None
    interventions: Optional[List[str]] = None

class EmotionSessionCreate(EmotionSessionBase):
    pass

class EmotionSession(EmotionSessionBase):
    id: int
    user_id: int
    start_time: datetime
    end_time: Optional[datetime] = None

    class Config:
        from_attributes = True

class RealtimeAnalysisResponse(BaseModel):
    frame: RealtimeFrame
    session: EmotionSession
    intervention: Optional[str] = None

class EmotionSummary(BaseModel):
    dominant_emotions: List[str]
    emotion_distribution: Dict[str, float]
    mood_trend: str
    suggested_interventions: List[str]
    next_steps: List[str]

class SessionSummaryResponse(BaseModel):
    session: EmotionSession
    summary: EmotionSummary
    interventions: List[str]

class VoiceAnalysis(BaseModel):
    arousal: float
    valence: float

class SpeechAnalysis(BaseModel):
    transcription: str
    sentiment: Dict[str, float]

class FacialAnalysis(BaseModel):
    average_emotions: Dict[str, float]
    dominant_emotion: str

class OverallAssessment(BaseModel):
    emotional_state: str
    confidence_score: float

class VideoAnalysisResult(BaseModel):
    facial_analysis: FacialAnalysis
    voice_analysis: VoiceAnalysis
    speech_analysis: SpeechAnalysis
    overall_assessment: OverallAssessment
    intervention: str

    class Config:
        orm_mode = True

class AudioFeatures(BaseModel):
    pitch_mean: float
    pitch_std: float
    energy: float
    tempo: float
    speech_rate: float
    pause_ratio: float
    voice_quality: float

class AudioEmotionAnalysis(BaseModel):
    arousal: float
    valence: float
    dominant_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]

class SpeechContent(BaseModel):
    transcription: str
    sentiment_score: float
    key_phrases: List[str]
    hesitation_count: int
    word_per_minute: float

class MentalStateIndicators(BaseModel):
    stress_level: float
    anxiety_level: float
    depression_indicators: List[str]
    mood_state: str
    energy_level: float
    coherence_score: float
    emotional_stability: float
    sleep_quality_indicator: float
    social_engagement_level: float
    cognitive_load: float
    resilience_score: float

class MentalHealthScores(BaseModel):
    anxiety_score: float
    depression_score: float
    stress_score: float
    emotional_regulation: float
    social_connection: float
    mindfulness: float
    sleep_quality: float
    cognitive_performance: float
    resilience: float
    life_satisfaction: float

class InterventionPlan(BaseModel):
    short_term: List[str]
    long_term: List[str]

class AudioAnalysisResponse(BaseModel):
    session_id: str
    timestamp: datetime
    audio_features: AudioFeatures
    emotion_analysis: AudioEmotionAnalysis
    speech_content: SpeechContent
    mental_state: MentalStateIndicators
    mental_health_scores: MentalHealthScores
    recommendations: List[str]
    follow_up_questions: List[str]
    risk_factors: Optional[List[str]] = None
    intervention_plan: InterventionPlan
    
    class Config:
        from_attributes = True

class EmotionTrendsResponse(BaseModel):
    daily_frequencies: Dict[date, Dict[str, int]]
    dominant_emotions: Dict[str, float]
    emotion_stability: float
    mood_variability: float
    positive_ratio: float

    class Config:
        from_attributes = True

class MentalHealthMetrics(BaseModel):
    stress_level: float
    anxiety_level: float
    mood_state: str
    emotional_stability: float
    social_engagement_level: float
    cognitive_load: float
    resilience_score: float

class StabilityMetric(BaseModel):
    average: float
    variance: float
    trend: str  # "improving" or "declining"

class MentalHealthTrendsResponse(BaseModel):
    daily_metrics: Dict[date, MentalHealthMetrics]
    overall_trends: Dict[str, List[float]]
    risk_factors: Dict[str, int]
    improvement_areas: List[str]
    stability_metrics: Dict[str, StabilityMetric]

    class Config:
        from_attributes = True

class EmotionalWellbeing(BaseModel):
    score: float
    stability: float

class MentalHealthStatus(BaseModel):
    stress_level: float
    anxiety_level: float
    emotional_stability: float

class WellnessReportResponse(BaseModel):
    overall_status: Dict[str, Union[EmotionalWellbeing, MentalHealthStatus, float]]
    trends: Dict[str, Union[EmotionTrendsResponse, MentalHealthTrendsResponse]]
    recommendations: List[str]
    risk_level: str  # "low", "moderate", or "high"

    class Config:
        from_attributes = True

class TextAnalysisRequest(BaseModel):
    content: str = Field(..., min_length=10, description="The text content to analyze (journal entry, thoughts, feelings)")
    context: Optional[str] = Field(None, description="Additional context about the entry (optional)")

class SentimentAnalysis(BaseModel):
    sentiment: Dict[str, Union[str, float]]
    emotions: Dict[str, Union[str, float]]

class LinguisticAnalysis(BaseModel):
    sentence_analysis: Dict[str, float]
    linguistic_features: Dict[str, int]
    vocabulary_diversity: float
    word_count: int

class ThemesAnalysis(BaseModel):
    main_themes: List[str]
    potential_concerns: Dict[str, int]

class MentalHealthAssessmentBase(BaseModel):
    depression_score: float
    anxiety_score: float
    stress_score: float
    sleep_quality_score: float
    emotional_regulation: float
    social_connection: float
    resilience_score: float
    mindfulness_score: float
    cognitive_metrics: Dict[str, Any]
    activity_data: Dict[str, Any]
    social_data: Dict[str, Any]
    sleep_data: Dict[str, Any]
    risk_factors: List[str]

class MentalHealthAssessmentCreate(MentalHealthAssessmentBase):
    pass

class MentalHealthAssessmentResponse(BaseModel):
    id: int
    user_id: int
    timestamp: datetime
    depression_score: float
    anxiety_score: float
    stress_score: float
    sleep_quality_score: float
    emotional_regulation: float
    social_connection: float
    resilience_score: float
    mindfulness_score: float
    cognitive_metrics: Dict = {}
    activity_data: Dict = {}
    social_data: Dict = {}
    sleep_data: Dict = {}
    risk_factors: List[str] = []
    cognitive_function: Dict = {}
    activity_level: float = 0.0
    social_engagement: float = 0.0
    sleep_patterns: Dict = {}
    suicide_risk: float = 0.0
    self_harm_risk: float = 0.0
    protective_factors: List[str] = []
    treatment_adherence: float = 0.0
    medication_compliance: float = 0.0
    therapy_attendance: float = 0.0
    progress_metrics: Dict = {}
    ai_insights: Dict = {
        "emotional_state": {},
        "key_concerns": [],
        "coping_mechanisms": [],
        "support_needs": [],
        "immediate_recommendations": []
    }
    recommended_interventions: List[Dict] = []
    predicted_outcomes: Dict = {}

    class Config:
        from_attributes = True

class MentalHealthInterventionBase(BaseModel):
    intervention_type: str
    description: str
    goals: List[str]
    expected_outcomes: Dict[str, Any]
    ai_recommendations: Dict[str, Any]
    personalized_approach: Dict[str, Any]

class MentalHealthInterventionCreate(MentalHealthInterventionBase):
    pass

class MentalHealthInterventionResponse(MentalHealthInterventionBase):
    id: int
    assessment_id: int
    timestamp: datetime
    progress_metrics: Dict[str, Any]
    adherence_score: float
    effectiveness_score: float

    class Config:
        orm_mode = True

class InterventionProgressCreate(BaseModel):
    intervention_id: int
    progress_metrics: Dict[str, Any]
    adherence_score: float
    effectiveness_score: float

class InterventionProgressResponse(BaseModel):
    intervention: MentalHealthInterventionResponse
    ai_feedback: Dict[str, Any]
    next_steps: List[Dict[str, Any]]

class MentalHealthTrendsResponse(BaseModel):
    trends: Dict[str, Dict[str, Any]]
    insights: Dict[str, Any]
    risk_level: str
    recommendations: List[Dict[str, Any]]

class TextAnalysisResponse(BaseModel):
    analysis_id: int
    timestamp: datetime
    sentiment_analysis: SentimentAnalysis
    linguistic_analysis: LinguisticAnalysis
    themes_analysis: ThemesAnalysis
    mental_health_assessment: MentalHealthAssessmentResponse
    personalized_interventions: List[Dict[str, Any]]

    class Config:
        from_attributes = True

# Stress Tracking Schemas
class StressTrackingBase(BaseModel):
    stress_level: float
    source: str
    context: str
    location: Optional[str] = None
    facial_analysis: Optional[Dict[str, Any]] = None
    voice_analysis: Optional[Dict[str, Any]] = None
    text_analysis: Optional[Dict[str, Any]] = None

class StressTrackingCreate(StressTrackingBase):
    pass

class StressTrackingResponse(StressTrackingBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Meditation Session Schemas
class MeditationSessionBase(BaseModel):
    session_type: str
    duration: int
    script: str
    audio_path: Optional[str] = None
    completion_status: Optional[float] = None
    user_feedback: Optional[Dict[str, Any]] = None
    effectiveness_score: Optional[float] = None

class MeditationSessionCreate(MeditationSessionBase):
    pass

class MeditationSessionResponse(MeditationSessionBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Mood Journal Schemas
class MoodJournalBase(BaseModel):
    text_content: Optional[str] = None
    audio_path: Optional[str] = None
    facial_emotions: Optional[Dict[str, Any]] = None
    mood_score: float
    dominant_emotions: List[str]
    sentiment_analysis: Dict[str, Any]
    themes: List[str]

class MoodJournalCreate(MoodJournalBase):
    pass

class MoodJournalResponse(MoodJournalBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Cognitive Game Schemas
class CognitiveGameBase(BaseModel):
    game_type: str
    difficulty_level: int
    duration: int
    score: float
    accuracy: float
    reaction_time: float
    completion_status: bool
    attention_score: float
    memory_score: float
    problem_solving_score: float

class CognitiveGameCreate(CognitiveGameBase):
    pass

class CognitiveGameResponse(CognitiveGameBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Sleep Record Schemas
class SleepRecordBase(BaseModel):
    date: date
    sleep_duration: float
    sleep_quality: float
    deep_sleep_duration: float
    rem_sleep_duration: float
    room_temperature: Optional[float] = None
    noise_level: Optional[float] = None
    light_level: Optional[float] = None
    bedtime_routine: Dict[str, Any]
    wake_up_time: datetime
    sleep_onset_time: datetime
    sleep_analysis: Dict[str, Any]
    recommendations: List[str]

class SleepRecordCreate(SleepRecordBase):
    pass

class SleepRecordResponse(SleepRecordBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True

# Therapy Session Schemas
class TherapySessionBase(BaseModel):
    session_type: str
    duration: int
    topic: str
    messages: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    key_concerns: List[str]
    session_summary: str
    action_items: List[str]
    follow_up_needed: bool
    escalation_level: int

class TherapySessionCreate(TherapySessionBase):
    pass

class TherapySessionResponse(TherapySessionBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Emergency Contact Schemas
class EmergencyContactBase(BaseModel):
    name: str
    relationship: str
    phone_number: str
    email: Optional[str] = None
    is_primary: bool = False
    notify_on_high_stress: bool = True
    notify_on_crisis: bool = True
    notify_on_missed_medication: bool = True

class EmergencyContactCreate(EmergencyContactBase):
    pass

class EmergencyContactResponse(EmergencyContactBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True

# Emergency Alert Schemas
class EmergencyAlertBase(BaseModel):
    alert_type: str
    severity: int
    description: str
    responded_by: Optional[str] = None
    response_time: Optional[datetime] = None
    resolution: Optional[str] = None

class EmergencyAlertCreate(EmergencyAlertBase):
    pass

class EmergencyAlertResponse(EmergencyAlertBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        orm_mode = True 