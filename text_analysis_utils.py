import os
from typing import Dict, List, Tuple, Optional
import logging
from transformers import pipeline
import numpy as np
from datetime import datetime
import requests
from dotenv import load_dotenv
import json
import re
from textblob import TextBlob
import spacy
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize models and NLP tools
try:
    # Sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Text classification for emotion detection
    emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    
    # Load spaCy model for linguistic analysis
    nlp = spacy.load("en_core_web_sm")
    
    # Mistral API setup
    MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    
    logger.info("All text analysis models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

def analyze_text_sentiment(text: str) -> Dict:
    """Perform detailed sentiment analysis on text."""
    try:
        # Basic sentiment using pipeline
        sentiment = sentiment_pipeline(text)[0]
        
        # Detailed sentiment using TextBlob
        blob = TextBlob(text)
        
        # Emotion analysis
        emotions = emotion_pipeline(text)
        
        return {
            "sentiment": {
                "label": sentiment["label"],
                "score": sentiment["score"],
                "polarity": float(blob.sentiment.polarity),
                "subjectivity": float(blob.sentiment.subjectivity)
            },
            "emotions": emotions[0]
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise

def analyze_linguistic_patterns(text: str) -> Dict:
    """Analyze linguistic patterns and writing style."""
    try:
        doc = nlp(text)
        
        # Analyze sentence structure
        sentence_lengths = [len(sent) for sent in doc.sents]
        
        # Count specific linguistic features
        features = {
            "question_marks": len([token for token in doc if token.text == "?"]),
            "exclamation_marks": len([token for token in doc if token.text == "!"]),
            "ellipsis": len([token for token in doc if token.text == "..."]),
            "negative_words": len([token for token in doc if token.dep_ == "neg"]),
            "personal_pronouns": len([token for token in doc if token.pos_ == "PRON" and token.text.lower() in ["i", "me", "my", "mine", "myself"]])
        }
        
        # Analyze vocabulary diversity
        words = [token.text.lower() for token in doc if token.is_alpha]
        vocabulary_diversity = len(set(words)) / len(words) if words else 0
        
        return {
            "sentence_analysis": {
                "avg_length": np.mean(sentence_lengths) if sentence_lengths else 0,
                "max_length": max(sentence_lengths) if sentence_lengths else 0,
                "num_sentences": len(sentence_lengths)
            },
            "linguistic_features": features,
            "vocabulary_diversity": float(vocabulary_diversity),
            "word_count": len(words)
        }
    except Exception as e:
        logger.error(f"Error in linguistic analysis: {str(e)}")
        raise

def extract_themes_and_concerns(text: str) -> Dict:
    """Extract main themes and potential mental health concerns."""
    try:
        doc = nlp(text)
        
        # Extract key phrases and themes
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Define mental health related keywords
        mental_health_keywords = {
            "anxiety": ["anxiety", "worried", "nervous", "stress", "fear", "panic"],
            "depression": ["depression", "sad", "hopeless", "worthless", "tired", "exhausted"],
            "trauma": ["trauma", "nightmare", "flashback", "scared", "hurt", "abuse"],
            "self_esteem": ["confidence", "self-esteem", "worthless", "failure", "ugly", "hate myself"],
            "relationships": ["relationship", "friend", "family", "lonely", "alone", "social"],
            "sleep": ["sleep", "insomnia", "nightmare", "tired", "rest", "exhausted"],
            "work_study": ["work", "study", "school", "college", "job", "career", "pressure"]
        }
        
        # Count occurrences of keywords in each category
        concerns = {}
        text_lower = text.lower()
        for category, keywords in mental_health_keywords.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            concerns[category] = count
        
        return {
            "main_themes": list(set(noun_phrases))[:10],
            "potential_concerns": concerns
        }
    except Exception as e:
        logger.error(f"Error extracting themes: {str(e)}")
        raise

async def generate_mental_health_assessment(
    sentiment_analysis: Dict,
    linguistic_analysis: Dict,
    themes_analysis: Dict
) -> Dict:
    """Generate comprehensive mental health assessment using Mistral."""
    try:
        # Prepare context for Mistral
        context = f"""As a mental health professional, analyze this writing assessment:

Sentiment Analysis:
- Overall sentiment: {sentiment_analysis['sentiment']['label']} (confidence: {sentiment_analysis['sentiment']['score']:.2f})
- Emotional polarity: {sentiment_analysis['sentiment']['polarity']:.2f}
- Subjectivity: {sentiment_analysis['sentiment']['subjectivity']:.2f}
- Detected emotions: {sentiment_analysis['emotions']}

Linguistic Patterns:
- Average sentence length: {linguistic_analysis['sentence_analysis']['avg_length']:.1f} words
- Use of personal pronouns: {linguistic_analysis['linguistic_features']['personal_pronouns']}
- Negative expressions: {linguistic_analysis['linguistic_features']['negative_words']}
- Question marks: {linguistic_analysis['linguistic_features']['question_marks']}
- Exclamation marks: {linguistic_analysis['linguistic_features']['exclamation_marks']}

Main Concerns:
{themes_analysis['potential_concerns']}

Please provide:
1. Mental health risk assessment (scale 0-10)
2. Key areas of concern
3. Emotional state analysis
4. Suggested coping strategies
5. Professional intervention recommendations
6. Immediate support suggestions

Format the response as a structured JSON with these categories."""

        # Call Mistral API
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json={"inputs": context}
        )
        
        if response.status_code != 200:
            logger.error(f"Mistral API error: {response.text}")
            return generate_default_assessment(sentiment_analysis, themes_analysis)

        # Parse response
        try:
            result = response.json()
            if isinstance(result, list):
                result = result[0]
            
            assessment = json.loads(result)
            return assessment
        except json.JSONDecodeError:
            logger.error("Failed to parse Mistral response")
            return generate_default_assessment(sentiment_analysis, themes_analysis)
            
    except Exception as e:
        logger.error(f"Error generating assessment: {str(e)}")
        return generate_default_assessment(sentiment_analysis, themes_analysis)

def generate_default_assessment(sentiment_analysis: Dict, themes_analysis: Dict) -> Dict:
    """Generate default assessment when Mistral API fails."""
    return {
        "risk_level": calculate_risk_level(sentiment_analysis, themes_analysis),
        "areas_of_concern": list(themes_analysis["potential_concerns"].keys()),
        "emotional_state": {
            "primary_emotion": sentiment_analysis["emotions"]["label"],
            "intensity": sentiment_analysis["emotions"]["score"],
            "stability": "moderate"
        },
        "coping_strategies": [
            "Practice deep breathing exercises",
            "Write in a journal regularly",
            "Engage in physical activity",
            "Maintain a regular sleep schedule",
            "Connect with supportive friends or family"
        ],
        "professional_help": {
            "recommended": sentiment_analysis["sentiment"]["polarity"] < -0.3,
            "type": ["Counseling", "Therapy"],
            "urgency": "moderate"
        },
        "immediate_support": [
            "Call a trusted friend or family member",
            "Use relaxation techniques",
            "Focus on present moment awareness",
            "Practice self-care activities"
        ]
    }

def calculate_risk_level(sentiment_analysis: Dict, themes_analysis: Dict) -> int:
    """Calculate risk level based on analysis results."""
    risk_score = 0
    
    # Add risk based on negative sentiment
    if sentiment_analysis["sentiment"]["polarity"] < 0:
        risk_score += abs(sentiment_analysis["sentiment"]["polarity"]) * 3
    
    # Add risk based on concerning themes
    for category, count in themes_analysis["potential_concerns"].items():
        if category in ["anxiety", "depression", "trauma"]:
            risk_score += count * 0.5
    
    # Cap the risk score at 10
    return min(int(risk_score), 10)

async def generate_personalized_interventions(assessment: Dict) -> Dict:
    """Generate personalized interventions based on assessment."""
    try:
        # Prepare context for Mistral
        context = f"""Based on this mental health assessment:
Risk Level: {assessment['risk_level']}/10
Primary Emotion: {assessment['emotional_state']['primary_emotion']}
Main Concerns: {', '.join(assessment['areas_of_concern'])}

Generate personalized interventions including:
1. Daily practices (5 specific activities)
2. Weekly goals (3 achievable goals)
3. Crisis management plan
4. Self-reflection prompts
5. Progress tracking metrics

Format as JSON with these categories."""

        # Call Mistral API
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json={"inputs": context}
        )
        
        if response.status_code != 200:
            return generate_default_interventions(assessment)

        # Parse response
        try:
            result = response.json()
            if isinstance(result, list):
                result = result[0]
            
            interventions = json.loads(result)
            return interventions
        except json.JSONDecodeError:
            return generate_default_interventions(assessment)
            
    except Exception as e:
        logger.error(f"Error generating interventions: {str(e)}")
        return generate_default_interventions(assessment)

def generate_default_interventions(assessment: Dict) -> Dict:
    """Generate default interventions when Mistral API fails."""
    return {
        "daily_practices": [
            "Morning meditation (5-10 minutes)",
            "Gratitude journaling",
            "Physical exercise (30 minutes)",
            "Mindful breathing exercises",
            "Evening reflection"
        ],
        "weekly_goals": [
            "Connect with at least two supportive people",
            "Complete three self-care activities",
            "Practice a new coping skill"
        ],
        "crisis_plan": {
            "immediate_steps": [
                "Use grounding techniques",
                "Contact support person",
                "Practice deep breathing"
            ],
            "emergency_contacts": [
                "Therapist/Counselor",
                "Crisis Hotline",
                "Trusted Friend/Family"
            ]
        },
        "reflection_prompts": [
            "What triggered my emotions today?",
            "What coping strategies worked well?",
            "What am I grateful for today?",
            "How did I show self-compassion?"
        ],
        "progress_metrics": [
            "Daily mood rating",
            "Sleep quality",
            "Anxiety level",
            "Social connection",
            "Coping skill usage"
        ]
    } 