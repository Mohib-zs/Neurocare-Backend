import cv2
import numpy as np
from deepface import DeepFace
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import deque
import json
import requests
import os
from dotenv import load_dotenv
import base64
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class EmotionFrame:
    frame_id: int
    emotions: Dict[str, float]
    dominant_emotion: str
    timestamp: datetime

class RealtimeEmotionAnalyzer:
    def __init__(self, window_size: int = 30, target_size: Tuple[int, int] = (224, 224)):
        self.window_size = window_size
        self.emotion_buffer = deque(maxlen=window_size)
        self.frame_counter = 0
        self.last_intervention_time = datetime.now()
        self.intervention_cooldown = timedelta(seconds=30)
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        self.target_size = target_size
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for emotion detection"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # Resize to target size
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image

    def process_frame(self, frame: np.ndarray, is_base64: bool = False) -> Dict[str, Any]:
        """Process a single frame and return structured analysis"""
        try:
            if is_base64:
                # Decode base64 image
                img_bytes = base64.b64decode(frame)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Preprocess image
            processed_frame = self.preprocess_image(frame)
            
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(
                processed_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            emotions = result[0]['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence_score = emotions[dominant_emotion] / 100.0
            
            # Create frame data
            self.frame_counter += 1
            frame_data = EmotionFrame(
                frame_id=self.frame_counter,
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                timestamp=datetime.now()
            )
            
            # Update emotion buffer
            self.emotion_buffer.append(frame_data)
            
            # Prepare response
            response = {
                "status": "success",
                "frame_id": frame_data.frame_id,
                "emotion_scores": emotions,
                "dominant_emotion": dominant_emotion,
                "confidence_score": confidence_score
            }
            
            # Add trend data if available
            if len(self.emotion_buffer) > 1:
                response["emotion_trend"] = self._get_emotion_trend()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "frame_id": self.frame_counter
            }

    def _get_emotion_trend(self) -> List[Dict[str, Any]]:
        """Get emotion trend from buffer"""
        return [
            {
                "frame": frame.frame_id,
                **frame.emotions
            }
            for frame in self.emotion_buffer
        ]

    def get_emotion_trends(self) -> Dict[str, List[float]]:
        """Get emotion trends over time"""
        if not self.emotion_buffer:
            return {}
            
        trends = {}
        for frame in self.emotion_buffer:
            for emotion, score in frame.emotions.items():
                if emotion not in trends:
                    trends[emotion] = []
                trends[emotion].append(score)
        return trends

    def get_rolling_average(self, window: int = 5) -> Dict[str, float]:
        """Calculate rolling average of emotions"""
        if not self.emotion_buffer or len(self.emotion_buffer) < window:
            return {}
            
        recent_frames = list(self.emotion_buffer)[-window:]
        avg_emotions = {}
        
        for emotion in recent_frames[0].emotions.keys():
            avg_emotions[emotion] = sum(
                frame.emotions[emotion] for frame in recent_frames
            ) / window
            
        return avg_emotions

    def should_trigger_intervention(self, threshold: float = 0.6) -> bool:
        """Check if intervention should be triggered based on emotional state"""
        if not self.should_generate_intervention():
            return False
            
        avg_emotions = self.get_rolling_average()
        if not avg_emotions:
            return False
            
        # Check for persistent distress signals
        distress_signals = {
            "sad": threshold,
            "angry": threshold,
            "fear": threshold
        }
        
        return any(
            avg_emotions.get(emotion, 0) >= threshold
            for emotion in distress_signals
        )

    def generate_realtime_intervention(self) -> Optional[str]:
        """Generate intervention based on real-time emotional state"""
        if not self.should_trigger_intervention():
            return None
            
        avg_emotions = self.get_rolling_average()
        if not avg_emotions:
            return None
            
        dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
        emotion_intensity = avg_emotions[dominant_emotion]
        
        # Create context-aware prompt
        prompt = self._create_intervention_prompt(dominant_emotion, emotion_intensity, avg_emotions)
        
        try:
            # Use ThreadPoolExecutor for non-blocking API call
            future = self.executor.submit(self._call_mistral_api, prompt)
            intervention = future.result(timeout=2.0)  # 2-second timeout
            
            if intervention:
                return self._post_process_intervention(intervention, dominant_emotion)
            return self._get_fallback_intervention(dominant_emotion)
            
        except Exception as e:
            logger.error(f"Error generating intervention: {str(e)}")
            return self._get_fallback_intervention(dominant_emotion)

    def _call_mistral_api(self, prompt: str) -> Optional[str]:
        """Make API call to Mistral"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": f"<s>[INST] {prompt} [/INST]",
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result[0].get('generated_text', '').strip()
            
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            return None

    def should_generate_intervention(self) -> bool:
        current_time = datetime.now()
        if current_time - self.last_intervention_time >= self.intervention_cooldown:
            self.last_intervention_time = current_time
            return True
        return False
        
    def _create_intervention_prompt(self, dominant_emotion: str, intensity: float, all_emotions: Dict[str, float]) -> str:
        """Creates a context-aware prompt for Mistral"""
        emotion_context = ", ".join([f"{k}: {v:.1f}%" for k, v in all_emotions.items()])
        
        base_prompt = f"""You are a supportive mental health assistant. The user's emotional state analysis shows:
Dominant emotion: {dominant_emotion} ({intensity:.1f}%)
Overall emotional state: {emotion_context}

Provide a brief, empathetic response (2-3 sentences) that:
1. Acknowledges their current emotional state
2. Offers a specific, actionable suggestion
3. Is supportive and encouraging

Keep the tone conversational and friendly. Do not use clinical language."""

        return base_prompt
    
    def _post_process_intervention(self, intervention: str, emotion: str) -> str:
        """Cleans and enhances Mistral's response"""
        # Remove any unwanted prefixes/suffixes
        prefixes_to_remove = [
            "Here's a supportive response:",
            "As an AI assistant,",
            "I understand",
            "I can see",
            "Let me provide"
        ]
        
        for prefix in prefixes_to_remove:
            if intervention.startswith(prefix):
                intervention = intervention[len(prefix):].strip()
        
        # Ensure the response isn't too long
        if len(intervention) > 150:
            intervention = ". ".join(intervention.split(". ")[:2]) + "."
            
        # Add emotion-specific emoji if the response doesn't have one
        emojis = {
            "happy": "😊",
            "sad": "💙",
            "angry": "🌱",
            "fear": "🤗",
            "surprise": "✨",
            "neutral": "💫"
        }
        
        if not any(emoji in intervention for emoji in emojis.values()):
            intervention = f"{emojis.get(emotion, '💫')} {intervention}"
            
        return intervention.strip()
            
    def _get_fallback_intervention(self, emotion: str) -> str:
        """Provides a fallback intervention when API call fails"""
        fallbacks = {
            "happy": "😊 Channel this positive energy into something creative or share your joy with others.",
            "sad": "💙 Take a moment for self-care and consider reaching out to someone you trust.",
            "angry": "🌱 Try taking three deep breaths and step away from the situation briefly.",
            "fear": "🤗 Ground yourself by focusing on five things you can see around you.",
            "surprise": "✨ Take a moment to process what's happening and write down your thoughts.",
            "neutral": "💫 This is a good time to check in with yourself and set a small goal."
        }
        return fallbacks.get(emotion, "💫 Take a moment to check in with yourself and your needs.")
    
    def generate_session_summary(self) -> Dict:
        if not self.emotion_buffer:
            return {}
            
        avg_emotions = self.get_rolling_average()
        emotion_trend = self.get_emotion_trends()
        
        # Get top 3 dominant emotions
        sorted_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = [emotion for emotion, _ in sorted_emotions[:3]]
        
        # Determine mood trend
        if len(self.emotion_buffer) > 1:
            first_half = list(self.emotion_buffer)[:len(self.emotion_buffer)//2]
            second_half = list(self.emotion_buffer)[len(self.emotion_buffer)//2:]
            
            first_avg = sum(frame.emotions[dominant_emotions[0]] for frame in first_half) / len(first_half)
            second_avg = sum(frame.emotions[dominant_emotions[0]] for frame in second_half) / len(second_half)
            
            if second_avg > first_avg:
                mood_trend = "improving"
            elif second_avg < first_avg:
                mood_trend = "declining"
            else:
                mood_trend = "stable"
        else:
            mood_trend = "insufficient data"
        
        return {
            "dominant_emotions": dominant_emotions,
            "emotion_distribution": avg_emotions,
            "mood_trend": mood_trend,
            "suggested_interventions": self._generate_intervention_suggestions(dominant_emotions[0]),
            "next_steps": self._generate_next_steps(dominant_emotions[0], mood_trend)
        }
    
    def _generate_intervention_suggestions(self, primary_emotion: str) -> List[str]:
        suggestions = {
            "happy": [
                "Channel this positive energy into a creative activity",
                "Share your joy with others through acts of kindness",
                "Document what's making you happy for future reflection"
            ],
            "sad": [
                "Practice gentle self-compassion exercises",
                "Reach out to a trusted friend or family member",
                "Engage in light physical activity to lift your mood"
            ],
            "angry": [
                "Try deep breathing exercises for 2-3 minutes",
                "Write down what's triggering your anger",
                "Take a brief walk to clear your mind"
            ],
            "fear": [
                "Ground yourself using the 5-4-3-2-1 sensory exercise",
                "Challenge anxious thoughts with evidence",
                "Practice progressive muscle relaxation"
            ],
            "surprise": [
                "Take a moment to process the unexpected",
                "Journal about what surprised you",
                "Consider if this reveals any new opportunities"
            ],
            "neutral": [
                "Set a small goal for the next hour",
                "Try a mindfulness exercise",
                "Engage in an activity you enjoy"
            ]
        }
        return suggestions.get(primary_emotion, suggestions["neutral"])
    
    def _generate_next_steps(self, primary_emotion: str, mood_trend: str) -> List[str]:
        base_steps = [
            "Continue monitoring your emotions",
            "Practice recommended interventions",
            "Track which activities influence your mood"
        ]
        
        if mood_trend == "declining" and primary_emotion in ["sad", "angry", "fear"]:
            base_steps.extend([
                "Consider scheduling a therapy session",
                "Increase self-care activities"
            ])
        elif mood_trend == "improving":
            base_steps.extend([
                "Note what's working well for you",
                "Build on positive activities"
            ])
            
        return base_steps

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False) 