import os
import cv2
import numpy as np
import librosa
import speech_recognition as sr
from typing import Dict, List, Tuple
import logging
from deepface import DeepFace
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)
from moviepy.editor import VideoFileClip
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
try:
    # Speech-to-text model
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Sentiment analysis for text
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Emotion analysis model
    emotion_analyzer = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None
    )
    
    logger.info("All video analysis models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing video analysis models: {str(e)}")
    raise

def save_uploaded_video(video_data: bytes, user_id: int) -> str:
    """Save uploaded video and return the file path."""
    upload_dir = "video_uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_{user_id}_video_{timestamp}.mp4"
    filepath = os.path.join(upload_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(video_data)
    
    return filepath

def extract_audio(video_path: str) -> str:
    """Extract audio from video file."""
    try:
        audio_path = video_path.replace('.mp4', '.wav')
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def analyze_facial_emotions(video_path: str) -> List[Dict[str, float]]:
    """Analyze facial emotions in video frames."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_emotions = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze every 30th frame (adjust as needed)
            if frame_count % 30 == 0:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotions = {
                    emotion: float(score)
                    for emotion, score in result[0]['emotion'].items()
                }
                frame_emotions.append(emotions)
            
            frame_count += 1
        
        cap.release()
        return frame_emotions
    except Exception as e:
        logger.error(f"Error analyzing facial emotions: {str(e)}")
        raise

def transcribe_audio(audio_path: str) -> str:
    """Transcribe speech from audio."""
    try:
        # Load audio
        audio, rate = librosa.load(audio_path, sr=16000)
        
        # Process through wav2vec
        inputs = wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = wav2vec_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = wav2vec_processor.batch_decode(predicted_ids)[0]
        
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise

def analyze_voice_emotions(audio_path: str) -> Dict[str, float]:
    """Analyze emotions in voice using audio features."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Extract audio features
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Calculate basic statistics of features
        features = {
            "energy": float(np.mean(librosa.feature.rms(y=y))),
            "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
            "pitch_std": float(np.std(librosa.yin(y, fmin=20, fmax=3000))),
            "spectral_contrast": float(np.mean(spectral_contrast))
        }
        
        # Map features to emotional indicators
        voice_emotions = {
            "arousal": (features["energy"] + features["tempo"]) / 2,
            "valence": (features["pitch_std"] + features["spectral_contrast"]) / 2
        }
        
        return voice_emotions
    except Exception as e:
        logger.error(f"Error analyzing voice emotions: {str(e)}")
        raise

def analyze_text_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment and emotions in transcribed text."""
    try:
        result = sentiment_analyzer(text)
        return {
            "sentiment": result[0]["label"],
            "confidence": float(result[0]["score"])
        }
    except Exception as e:
        logger.error(f"Error analyzing text sentiment: {str(e)}")
        raise

def generate_comprehensive_analysis(
    facial_emotions: List[Dict[str, float]],
    voice_emotions: Dict[str, float],
    text_sentiment: Dict[str, float],
    transcription: str
) -> Dict:
    """Generate comprehensive emotional analysis from all sources."""
    
    # Average facial emotions across frames
    avg_facial_emotions = {}
    for emotion in facial_emotions[0].keys():
        avg_facial_emotions[emotion] = float(np.mean([frame[emotion] for frame in facial_emotions]))
    
    # Determine dominant emotions
    dominant_facial = max(avg_facial_emotions.items(), key=lambda x: x[1])[0]
    
    # Combine all analyses
    comprehensive_analysis = {
        "facial_analysis": {
            "average_emotions": avg_facial_emotions,
            "dominant_emotion": dominant_facial
        },
        "voice_analysis": voice_emotions,
        "speech_analysis": {
            "transcription": transcription,
            "sentiment": text_sentiment
        },
        "overall_assessment": {
            "emotional_state": dominant_facial,
            "confidence_score": float(np.mean([
                max(avg_facial_emotions.values()),
                voice_emotions["arousal"],
                text_sentiment["confidence"]
            ]))
        }
    }
    
    return comprehensive_analysis

def generate_mental_health_intervention(analysis: Dict) -> str:
    """Generate mental health intervention based on analysis."""
    try:
        # Extract key information
        dominant_emotion = analysis['facial_analysis']['dominant_emotion']
        voice_arousal = analysis['voice_analysis']['arousal']
        text_sentiment = analysis['speech_analysis']['sentiment']
        
        # Generate intervention based on the analysis
        intervention = f"""Based on the analysis:

1. Primary Emotional State: {dominant_emotion}
2. Voice Analysis: {'High' if voice_arousal > 0.5 else 'Low'} arousal level
3. Speech Sentiment: {text_sentiment['sentiment']} with {text_sentiment['confidence']:.2f} confidence

Recommendations:
1. Consider practicing mindfulness or relaxation techniques
2. Maintain a regular sleep schedule
3. Engage in physical activity
4. Connect with supportive friends or family
5. Consider seeking professional help if these feelings persist

Remember that emotional states are temporary and seeking help is a sign of strength."""
        
        return intervention
    except Exception as e:
        logger.error(f"Error generating mental health intervention: {str(e)}")
        return "Based on the analysis, I recommend seeking professional mental health support for a comprehensive assessment." 