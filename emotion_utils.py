import os
from datetime import datetime
import base64
from deepface import DeepFace
import cv2
import numpy as np
import logging
from typing import Dict, Tuple
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Hugging Face token
try:
    hf_token = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_token:
        logger.warning("HUGGINGFACE_API_KEY not found, using default suggestions")
    else:
        logger.info("Using Hugging Face API for model inference")
except Exception as e:
    logger.error(f"Error initializing Hugging Face token: {str(e)}")
    hf_token = None

def save_uploaded_image(image_data: bytes, user_id: int) -> str:
    """Save uploaded image and return the file path."""
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_{user_id}_{timestamp}.jpg"
    filepath = os.path.join(upload_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    return filepath

def analyze_emotions(image_path: str) -> Tuple[Dict[str, float], str]:
    """Analyze emotions in image using DeepFace."""
    try:
        result = DeepFace.analyze(image_path, actions=['emotion'])
        # Convert numpy float32 to regular Python float
        emotions = {
            emotion: float(score)
            for emotion, score in result[0]['emotion'].items()
        }
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        return emotions, dominant_emotion
    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        raise

def get_suggestion(dominant_emotion: str) -> str:
    """Get a simple suggestion based on the dominant emotion."""
    suggestions = {
        "angry": "Take a few deep breaths and count to 10. Try to identify what triggered your anger.",
        "disgust": "Try to understand what caused this feeling. Sometimes our initial reactions can be strong.",
        "fear": "Remember that fear is a natural response. Try to break down what you're afraid of into smaller, manageable parts.",
        "happy": "Great to see you're feeling happy! Try to share this positive energy with others.",
        "sad": "It's okay to feel sad. Try talking to someone you trust about how you're feeling.",
        "surprise": "Take a moment to process what surprised you. Sometimes unexpected events can be opportunities.",
        "neutral": "You seem calm and composed. This is a good state for making clear decisions."
    }
    return suggestions.get(dominant_emotion, "Take a moment to reflect on your current emotional state.")

def generate_mental_health_intervention(emotions: Dict[str, float], dominant_emotion: str) -> str:
    """Generate a detailed intervention using Mistral model if available."""
    if hf_token is None:
        return get_suggestion(dominant_emotion)

    try:
        # Create a detailed prompt for the model
        prompt = f"""You are a supportive mental health assistant. The user's emotional state analysis shows:
Dominant emotion: {dominant_emotion}
Emotional distribution: {', '.join([f'{k}: {v:.1f}%' for k, v in emotions.items()])}

Provide a brief, empathetic response (2-3 sentences) that:
1. Acknowledges their current emotional state
2. Offers a specific, actionable suggestion
3. Is supportive and encouraging

Keep the tone conversational and friendly. Do not use clinical language."""

        # Generate response with the model
        response = call_mistral_api(prompt)
        return response

    except Exception as e:
        logger.error(f"Error generating intervention: {str(e)}")
        return get_suggestion(dominant_emotion)

def process_base64_image(base64_string: str) -> bytes:
    """Convert base64 string to bytes."""
    try:
        # Remove potential header
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise

def call_mistral_api(prompt: str) -> str:
    """Make API call to Mistral model"""
    if not hf_token:
        return "Default intervention suggestion: Consider practicing mindfulness and reaching out to supportive friends or family."
    
    try:
        headers = {
            "Authorization": f"Bearer {hf_token}",
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
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result[0].get('generated_text', '').strip()
        
    except Exception as e:
        logger.error(f"API call error: {str(e)}")
        return "Default intervention suggestion: Consider practicing mindfulness and reaching out to supportive friends or family." 