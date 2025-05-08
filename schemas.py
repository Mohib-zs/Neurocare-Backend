from pydantic import BaseModel, EmailStr, constr, Field
from typing import Optional, List, Dict, Union
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

class MentalHealthAssessment(BaseModel):
    risk_level: int
    areas_of_concern: List[str]
    emotional_state: Dict[str, Union[str, float]]
    coping_strategies: List[str]
    professional_help: Dict[str, Union[bool, List[str], str]]
    immediate_support: List[str]

class PersonalizedInterventions(BaseModel):
    daily_practices: List[str]
    weekly_goals: List[str]
    crisis_plan: Dict[str, List[str]]
    reflection_prompts: List[str]
    progress_metrics: List[str]

class TextAnalysisResponse(BaseModel):
    analysis_id: int
    timestamp: datetime
    sentiment_analysis: SentimentAnalysis
    linguistic_analysis: LinguisticAnalysis
    themes_analysis: ThemesAnalysis
    mental_health_assessment: MentalHealthAssessment
    personalized_interventions: PersonalizedInterventions

    class Config:
        from_attributes = True 