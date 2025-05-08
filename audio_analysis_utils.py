import os
import uuid
import librosa
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from transformers import pipeline
import soundfile as sf
from scipy.stats import zscore
import requests
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize models
try:
    # Speech-to-text model
    asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    
    # Sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Text generation for recommendations (using Mistral API)
    MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    
    logger.info("All audio analysis models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

def save_audio_file(audio_data: bytes, user_id: int) -> str:
    """Save uploaded audio file and return the file path."""
    upload_dir = "audio_uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_{user_id}_audio_{timestamp}.wav"
    filepath = os.path.join(upload_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(audio_data)
    
    return filepath

def extract_audio_features(audio_path: str) -> Dict:
    """Extract acoustic features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitch[pitch > 0])
        pitch_std = np.std(pitch[pitch > 0])
        
        # Energy and tempo
        energy = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Speech rate and pauses
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        speech_rate = np.mean(pulse)
        pause_ratio = np.sum(pulse < np.mean(pulse)) / len(pulse)
        
        # Voice quality (using spectral features)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        voice_quality = spectral_centroid / sr
        
        return {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "energy": float(energy),
            "tempo": float(tempo),
            "speech_rate": float(speech_rate),
            "pause_ratio": float(pause_ratio),
            "voice_quality": float(voice_quality)
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        raise

def analyze_voice_emotion(audio_features: Dict) -> Dict:
    """Analyze emotional content from voice features."""
    try:
        # Normalize features
        features = np.array([
            audio_features["pitch_mean"],
            audio_features["pitch_std"],
            audio_features["energy"],
            audio_features["tempo"],
            audio_features["speech_rate"]
        ])
        normalized_features = zscore(features)
        
        # Calculate arousal (energy/intensity)
        arousal = (normalized_features[2] + normalized_features[3]) / 2
        
        # Calculate valence (emotional positivity/negativity)
        valence = (normalized_features[0] - normalized_features[1] + normalized_features[4]) / 3
        
        # Map to emotions based on arousal-valence space
        emotions = {
            "happy": max(0, (arousal + valence) / 2),
            "sad": max(0, (-arousal - valence) / 2),
            "angry": max(0, (arousal - valence) / 2),
            "calm": max(0, (-arousal + valence) / 2),
            "anxious": max(0, arousal * (1 - abs(valence))),
            "neutral": max(0, 1 - abs(arousal) - abs(valence))
        }
        
        # Get dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        confidence = emotions[dominant_emotion]
        
        return {
            "arousal": float(arousal),
            "valence": float(valence),
            "dominant_emotion": dominant_emotion,
            "confidence": float(confidence),
            "emotion_scores": {k: float(v) for k, v in emotions.items()}
        }
    except Exception as e:
        logger.error(f"Error analyzing voice emotion: {str(e)}")
        raise

def transcribe_and_analyze_speech(audio_path: str) -> Dict:
    """Transcribe speech and analyze content."""
    try:
        # Transcribe audio
        transcription = asr_pipeline(audio_path)[0]["text"]
        
        # Analyze sentiment
        sentiment = sentiment_pipeline(transcription)[0]
        sentiment_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]
        
        # Calculate speech metrics
        words = transcription.split()
        duration = librosa.get_duration(filename=audio_path)
        word_per_minute = (len(words) / duration) * 60
        
        # Count hesitations (um, uh, etc.)
        hesitation_words = ["um", "uh", "er", "ah", "like", "you know"]
        hesitation_count = sum(1 for word in words if word.lower() in hesitation_words)
        
        # Extract key phrases (simple implementation)
        key_phrases = [" ".join(words[i:i+3]) for i in range(0, len(words)-2, 3)][:5]
        
        return {
            "transcription": transcription,
            "sentiment_score": float(sentiment_score),
            "key_phrases": key_phrases,
            "hesitation_count": hesitation_count,
            "word_per_minute": float(word_per_minute)
        }
    except Exception as e:
        logger.error(f"Error in speech analysis: {str(e)}")
        raise

def assess_mental_state(
    audio_features: Dict,
    emotion_analysis: Dict,
    speech_content: Dict
) -> Dict:
    """Assess mental state with enhanced metrics using audio analysis."""
    try:
        # Calculate base metrics
        stress_level = (
            emotion_analysis["arousal"] * 0.4 +
            (1 - audio_features["voice_quality"]) * 0.3 +
            (speech_content["hesitation_count"] / 100) * 0.3
        )
        
        # Calculate anxiety level
        anxiety_level = (
            emotion_analysis["emotion_scores"].get("anxious", 0) * 0.4 +
            (speech_content["word_per_minute"] / 180) * 0.3 +
            (1 - audio_features["pause_ratio"]) * 0.3
        )
        
        # Identify depression indicators
        depression_indicators = []
        if emotion_analysis["valence"] < -0.4:
            depression_indicators.append("Negative emotional valence")
        if audio_features["energy"] < 0.3:
            depression_indicators.append("Low energy in voice")
        if speech_content["word_per_minute"] < 100:
            depression_indicators.append("Slow speech pattern")
            
        # Calculate additional metrics
        sleep_quality_indicator = 1.0 - (
            abs(audio_features["pitch_std"]) * 0.3 +
            (1 - audio_features["voice_quality"]) * 0.4 +
            (stress_level > 0.7) * 0.3
        )
        
        social_engagement_level = (
            audio_features["energy"] * 0.3 +
            emotion_analysis["arousal"] * 0.3 +
            (1 - abs(emotion_analysis["valence"])) * 0.4
        )
        
        cognitive_load = (
            speech_content["hesitation_count"] / 20 +
            (1 - audio_features["coherence_score"]) +
            (stress_level > 0.6)
        ) / 3
        
        resilience_score = 1.0 - (
            stress_level * 0.3 +
            anxiety_level * 0.3 +
            cognitive_load * 0.4
        )
        
        return {
            "stress_level": float(stress_level),
            "anxiety_level": float(anxiety_level),
            "depression_indicators": depression_indicators,
            "mood_state": emotion_analysis["dominant_emotion"],
            "energy_level": float(audio_features["energy"]),
            "coherence_score": float(audio_features["coherence_score"]),
            "emotional_stability": float(emotion_analysis["confidence"]),
            "sleep_quality_indicator": float(sleep_quality_indicator),
            "social_engagement_level": float(social_engagement_level),
            "cognitive_load": float(cognitive_load),
            "resilience_score": float(resilience_score)
        }
    except Exception as e:
        logger.error(f"Error assessing mental state: {str(e)}")
        raise

def generate_mental_health_scores(mental_state: Dict) -> Dict:
    """Generate detailed mental health scores using Mistral."""
    try:
        prompt = f"""As a mental health professional, analyze these indicators and provide detailed mental health scores:

Mental State Indicators:
- Stress Level: {mental_state['stress_level']:.2f}
- Anxiety Level: {mental_state['anxiety_level']:.2f}
- Depression Indicators: {', '.join(mental_state['depression_indicators'])}
- Mood: {mental_state['mood_state']}
- Energy Level: {mental_state['energy_level']:.2f}
- Sleep Quality: {mental_state['sleep_quality_indicator']:.2f}
- Social Engagement: {mental_state['social_engagement_level']:.2f}
- Cognitive Load: {mental_state['cognitive_load']:.2f}
- Resilience: {mental_state['resilience_score']:.2f}

Please provide numerical scores (0-1) for:
1. Anxiety
2. Depression
3. Stress
4. Emotional Regulation
5. Social Connection
6. Mindfulness
7. Sleep Quality
8. Cognitive Performance
9. Resilience
10. Life Satisfaction

Also provide a brief explanation for each score."""

        # Call Mistral API
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        response.raise_for_status()
        
        # Parse response and extract scores
        result = response.json()[0]["generated_text"]
        scores = {}
        
        # Parse the scores from the response
        score_mapping = {
            "anxiety": "anxiety_score",
            "depression": "depression_score",
            "stress": "stress_score",
            "emotional regulation": "emotional_regulation",
            "social connection": "social_connection",
            "mindfulness": "mindfulness",
            "sleep quality": "sleep_quality",
            "cognitive performance": "cognitive_performance",
            "resilience": "resilience",
            "life satisfaction": "life_satisfaction"
        }
        
        for line in result.split("\n"):
            for key, score_key in score_mapping.items():
                if key.lower() in line.lower():
                    try:
                        score = float(re.search(r"(\d+\.?\d*)", line).group(1))
                        scores[score_key] = min(max(score, 0), 1)  # Ensure score is between 0 and 1
                    except:
                        continue
        
        return scores
        
    except Exception as e:
        logger.error(f"Error generating mental health scores: {str(e)}")
        return {
            "anxiety_score": mental_state["anxiety_level"],
            "depression_score": 0.5,  # Default score
            "stress_score": mental_state["stress_level"],
            "emotional_regulation": mental_state["emotional_stability"],
            "social_connection": mental_state["social_engagement_level"],
            "mindfulness": 1 - mental_state["cognitive_load"],
            "sleep_quality": mental_state["sleep_quality_indicator"],
            "cognitive_performance": 1 - mental_state["cognitive_load"],
            "resilience": mental_state["resilience_score"],
            "life_satisfaction": (1 - mental_state["stress_level"] + mental_state["resilience_score"]) / 2
        }

async def generate_recommendations(
    emotion_analysis: Dict,
    mental_state: Dict,
    mental_health_scores: Dict
) -> Tuple[List[str], List[str]]:
    """Generate personalized recommendations and follow-up questions."""
    try:
        # Prepare the context for Mistral
        context = f"""Based on the following mental health assessment:
- Dominant emotion: {emotion_analysis['dominant_emotion']}
- Stress level: {mental_state['stress_level']:.2f}
- Anxiety level: {mental_state['anxiety_level']:.2f}
- Depression indicators: {', '.join(mental_state['depression_indicators'])}
- Emotional stability: {mental_state['emotional_stability']:.2f}
- Social engagement: {mental_state['social_engagement_level']:.2f}

Generate:
1. Five specific, actionable recommendations for improving mental well-being
2. Three empathetic follow-up questions to better understand the person's state

Format the response as a JSON with 'recommendations' and 'follow_up_questions' lists."""

        # Prepare headers for Hugging Face API
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }

        # Make API request to Mistral
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json={"inputs": context}
        )

        if response.status_code != 200:
            logger.error(f"Mistral API error: {response.text}")
            # Fallback to default recommendations if API fails
            return generate_default_recommendations(mental_state)

        # Parse the response
        try:
            result = response.json()
            if isinstance(result, list):
                result = result[0]  # Get first generation if multiple are returned
            
            # Extract recommendations and questions from the response
            # The model should return a JSON string that we can parse
            import json
            parsed_response = json.loads(result)
            
            recommendations = parsed_response.get("recommendations", [])
            follow_up_questions = parsed_response.get("follow_up_questions", [])

            # Ensure we have at least some recommendations
            if not recommendations:
                return generate_default_recommendations(mental_state)

            return recommendations, follow_up_questions

        except json.JSONDecodeError:
            logger.error("Failed to parse Mistral response as JSON")
            return generate_default_recommendations(mental_state)

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return generate_default_recommendations(mental_state)

def generate_default_recommendations(mental_state: Dict) -> Tuple[List[str], List[str]]:
    """Generate default recommendations when Mistral API is unavailable."""
    recommendations = [
        "Practice deep breathing exercises for 5 minutes each morning",
        "Take a 15-minute walk outside daily",
        "Maintain a regular sleep schedule",
        "Connect with a friend or family member today",
        "Write down three things you're grateful for"
    ]
    
    follow_up_questions = [
        "How have you been sleeping lately?",
        "What activities help you feel most relaxed?",
        "Would you like to talk more about what's on your mind?"
    ]
    
    return recommendations, follow_up_questions 