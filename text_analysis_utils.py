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
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    
    # Together API setup
    TOGETHER_API_URL = "https://router.huggingface.co/together/v1/chat/completions"
    TOGETHER_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    if not TOGETHER_API_KEY:
        logger.error("HUGGINGFACE_API_KEY environment variable is not set. Please set it with your Together API key.")
        raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
    )
    
    # Create a session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
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

async def call_mistral_api(prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Call Together API with retry logic."""
    if not TOGETHER_API_KEY:
        logger.error("HUGGINGFACE_API_KEY environment variable is not set")
        return None

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a mental health assessment AI. You must provide specific numerical scores (0-1) for each mental health metric based on the text analysis. Do not return default or zero values unless absolutely certain there are no indicators."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "temperature": 0.3,  # Lower temperature for more consistent scoring
        "max_tokens": 1500,
        "top_p": 0.95,
        "response_format": { "type": "json_object" }
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting API call (attempt {attempt + 1}/{max_retries})")
            logger.info(f"API URL: {TOGETHER_API_URL}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = session.post(
                TOGETHER_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Raw API response: {json.dumps(result, indent=2)}")
                return result
            elif response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = 20 * (attempt + 1)  # Increase wait time with each retry
                logger.info(f"Model is loading, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"API error: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retrying
                    continue
                return None
                
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
    
    return None

async def generate_mental_health_assessment(
    sentiment_analysis: Dict,
    linguistic_analysis: Dict,
    themes_analysis: Dict,
    original_text: str
) -> Dict:
    """Generate comprehensive mental health assessment using Mistral."""
    try:
        # Prepare context for Mistral
        prompt = f"""Analyze this person's writing and provide a detailed mental health assessment with specific numerical scores.

Original Text:
"{original_text}"

Analysis Results:
1. Sentiment Analysis:
- Overall sentiment: {sentiment_analysis['sentiment']['label']} (confidence: {sentiment_analysis['sentiment']['score']:.2f})
- Emotional polarity: {sentiment_analysis['sentiment']['polarity']:.2f}
- Subjectivity: {sentiment_analysis['sentiment']['subjectivity']:.2f}
- Detected emotions: {sentiment_analysis['emotions']}

2. Writing Style Analysis:
- Average sentence length: {linguistic_analysis['sentence_analysis']['avg_length']:.1f} words
   - Personal pronouns used: {linguistic_analysis['linguistic_features']['personal_pronouns']}
- Negative expressions: {linguistic_analysis['linguistic_features']['negative_words']}
- Question marks: {linguistic_analysis['linguistic_features']['question_marks']}
- Exclamation marks: {linguistic_analysis['linguistic_features']['exclamation_marks']}

3. Identified Themes and Concerns:
   {json.dumps(themes_analysis['potential_concerns'], indent=2)}

Based on the above analysis, provide a detailed mental health assessment in JSON format. You MUST provide specific numerical scores (0-1) for each metric based on the text analysis. Do not return default or zero values unless absolutely certain there are no indicators.

Required fields with scoring guidelines:
1. Core Mental Health Scores (0-1 scale, where 0 is optimal and 1 is severe):
   - depression_score: Based on expressions of sadness, hopelessness, lack of interest
   - anxiety_score: Based on expressions of worry, fear, racing thoughts
   - stress_score: Based on expressions of pressure, overwhelm, tension
   - sleep_quality_score: Based on mentions of sleep issues, fatigue, restlessness
   - emotional_regulation: Based on ability to manage emotions, mood stability
   - social_connection: Based on social engagement, relationships, isolation
   - resilience_score: Based on coping ability, adaptability, recovery
   - mindfulness_score: Based on present-moment awareness, clarity of thought

2. Risk Assessment:
   - suicide_risk: 0-1 scale based on expressions of hopelessness, worthlessness
   - self_harm_risk: 0-1 scale based on expressions of self-harm ideation
   - risk_factors: List of specific risk factors identified
   - protective_factors: List of protective factors identified

3. Treatment and Progress:
   - treatment_adherence: 0-1 scale based on engagement with treatment
   - medication_compliance: 0-1 scale based on medication management
   - therapy_attendance: 0-1 scale based on therapy engagement

4. Detailed Analysis:
   - emotional_state: {
       "primary_emotion": string,
       "intensity": float (0-1),
       "stability": string ("stable", "moderate", "unstable")
   }
   - key_concerns: List of specific concerns identified
   - coping_mechanisms: List of current coping strategies
   - support_needs: List of support requirements
   - immediate_recommendations: List of urgent actions needed

5. Additional Metrics:
   - cognitive_function: {
       "clarity": float (0-1),
       "concentration": float (0-1),
       "memory": float (0-1)
   }
   - activity_level: float (0-1) based on energy and engagement
   - social_engagement: float (0-1) based on social interaction
   - sleep_patterns: {
       "quality": float (0-1),
       "consistency": float (0-1),
       "disturbances": List[string]
   }

Return ONLY the JSON object with these fields, nothing else. You MUST provide specific numerical scores based on the text analysis. Do not return default or zero values unless absolutely certain there are no indicators."""

        # Call Together API with retry logic
        result = await call_mistral_api(prompt)
        if not result:
            logger.error("Failed to get response from Together API after retries")
            return generate_default_assessment(sentiment_analysis, themes_analysis)

        # Parse response
        try:
            # Extract the message content from the response
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"Generated text: {response_text}")
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    assessment = json.loads(json_match.group())
                    logger.info(f"Parsed assessment: {json.dumps(assessment, indent=2)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from response: {str(e)}")
                    logger.error(f"JSON content: {json_match.group()}")
                    return generate_default_assessment(sentiment_analysis, themes_analysis)
                
                # Create the assessment structure matching the database schema
                final_assessment = {
                    "depression_score": float(assessment.get("depression_score", 0.0)),
                    "anxiety_score": float(assessment.get("anxiety_score", 0.0)),
                    "stress_score": float(assessment.get("stress_score", 0.0)),
                    "sleep_quality_score": float(assessment.get("sleep_quality_score", 0.0)),
                    "emotional_regulation": float(assessment.get("emotional_regulation", 0.0)),
                    "social_connection": float(assessment.get("social_connection", 0.0)),
                    "resilience_score": float(assessment.get("resilience_score", 0.0)),
                    "mindfulness_score": float(assessment.get("mindfulness_score", 0.0)),
                    "cognitive_metrics": assessment.get("cognitive_function", {
                        "clarity": 0.0,
                        "concentration": 0.0,
                        "memory": 0.0
                    }),
                    "activity_data": {
                        "level": float(assessment.get("activity_level", 0.0))
                    },
                    "social_data": {
                        "engagement": float(assessment.get("social_engagement", 0.0))
                    },
                    "sleep_data": assessment.get("sleep_patterns", {
                        "quality": 0.0,
                        "consistency": 0.0,
                        "disturbances": []
                    }),
                    "risk_factors": assessment.get("risk_factors", []),
                    "cognitive_function": {
                        "clarity": float(assessment.get("cognitive_function", {}).get("clarity", 0.0)),
                        "concentration": float(assessment.get("cognitive_function", {}).get("concentration", 0.0)),
                        "memory": float(assessment.get("cognitive_function", {}).get("memory", 0.0))
                    },
                    "activity_level": float(assessment.get("activity_level", 0.0)),
                    "social_engagement": float(assessment.get("social_engagement", 0.0)),
                    "sleep_patterns": assessment.get("sleep_patterns", {
                        "quality": 0.0,
                        "consistency": 0.0,
                        "disturbances": []
                    }),
                    "suicide_risk": float(assessment.get("suicide_risk", 0.0)),
                    "self_harm_risk": float(assessment.get("self_harm_risk", 0.0)),
                    "protective_factors": assessment.get("protective_factors", []),
                    "treatment_adherence": float(assessment.get("treatment_adherence", 0.0)),
                    "medication_compliance": float(assessment.get("medication_compliance", 0.0)),
                    "therapy_attendance": float(assessment.get("therapy_attendance", 0.0)),
                    "progress_metrics": {},
                    "ai_insights": {
                        "emotional_state": {
                            "primary_emotion": sentiment_analysis["emotions"]["label"],
                            "intensity": float(sentiment_analysis["emotions"]["score"]),
                            "stability": "moderate"
                        },
                        "key_concerns": assessment.get("key_concerns", []),
                        "coping_mechanisms": assessment.get("coping_mechanisms", []),
                        "support_needs": assessment.get("support_needs", []),
                        "immediate_recommendations": assessment.get("immediate_recommendations", ["Please contact a mental health professional for support"])
                    }
                }
                
                # Log the final assessment
                logger.info(f"Final assessment with calculated scores: {json.dumps(final_assessment, indent=2)}")
                return final_assessment
            else:
                logger.error("No JSON object found in response")
                logger.error(f"Full response text: {response_text}")
                return generate_default_assessment(sentiment_analysis, themes_analysis)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            logger.error(f"Response content: {response_text}")
            return generate_default_assessment(sentiment_analysis, themes_analysis)
            
    except Exception as e:
        logger.error(f"Error generating assessment: {str(e)}")
        logger.error(f"Full error details: {e.__dict__}")
        return generate_default_assessment(sentiment_analysis, themes_analysis)

def generate_default_assessment(sentiment_analysis: Dict, themes_analysis: Dict) -> Dict:
    """Generate default assessment when Together API fails."""
    logger.warning("Using default assessment due to API failure")
    return {
        "depression_score": 0.0,
        "anxiety_score": 0.0,
        "stress_score": 0.0,
        "sleep_quality_score": 0.0,
        "emotional_regulation": 0.0,
        "social_connection": 0.0,
        "resilience_score": 0.0,
        "mindfulness_score": 0.0,
        "suicide_risk": 0.0,
        "self_harm_risk": 0.0,
        "risk_factors": [],
        "protective_factors": [],
        "treatment_adherence": 0.0,
        "medication_compliance": 0.0,
        "therapy_attendance": 0.0,
        "cognitive_function": {
            "clarity": 0.0,
            "concentration": 0.0,
            "memory": 0.0
        },
        "activity_level": 0.0,
        "social_engagement": 0.0,
        "sleep_patterns": {
            "quality": 0.0,
            "consistency": 0.0,
            "disturbances": []
        },
        "emotional_state": {
            "primary_emotion": sentiment_analysis["emotions"]["label"],
            "intensity": sentiment_analysis["emotions"]["score"],
            "stability": "moderate"
        },
        "key_concerns": list(themes_analysis["potential_concerns"].keys()),
        "coping_mechanisms": [],
        "support_needs": [],
        "immediate_recommendations": ["Please contact a mental health professional for support"]
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

async def generate_personalized_interventions(assessment: Dict, original_text: str) -> List[Dict]:
    """Generate personalized interventions using Mistral AI based on the user's text and assessment."""
    try:
        # Verify API token
        if not TOGETHER_API_KEY:
            logger.error("TOGETHER_API_KEY is not set")
            raise ValueError("TOGETHER_API_KEY is not set")

        # Prepare context for Mistral
        prompt = f"""You are a mental health professional. Based on this person's writing and assessment, generate personalized interventions.

Original Text:
"{original_text}"

Assessment Summary:
- Depression Score: {assessment.get('depression_score', 0)}
- Anxiety Score: {assessment.get('anxiety_score', 0)}
- Stress Score: {assessment.get('stress_score', 0)}
- Sleep Quality: {assessment.get('sleep_quality_score', 0)}
- Emotional State: {json.dumps(assessment.get('emotional_state', {}), indent=2)}
- Key Concerns: {json.dumps(assessment.get('key_concerns', []), indent=2)}
- Risk Factors: {json.dumps(assessment.get('risk_factors', []), indent=2)}
- Protective Factors: {json.dumps(assessment.get('protective_factors', []), indent=2)}

Generate a list of 3-5 personalized interventions in JSON format. Each intervention should be a JSON object with these fields:
- type: string (category of intervention)
- description: string (detailed intervention description)
- duration: string (how long to practice)
- frequency: string (how often to practice)
- expected_outcome: string (what to expect)
- rationale: string (why this intervention is recommended for their specific situation)

Focus on evidence-based interventions that directly address their expressed concerns and needs. Be specific and reference details from their writing.

Return ONLY the JSON array of interventions, nothing else."""

        # Call Together API
        result = await call_mistral_api(prompt)
        if not result:
            logger.error("Failed to get response from Together API")
            return generate_default_interventions(assessment)

        # Parse response
        try:
            # Extract the message content from the response
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"Generated text: {response_text}")
            
            # Try to find JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                interventions = json.loads(json_match.group())
                logger.info(f"Parsed interventions: {json.dumps(interventions, indent=2)}")
                
                # Validate intervention structure
                for intervention in interventions:
                    intervention.setdefault("type", "general")
                    intervention.setdefault("description", "Please consult with a mental health professional")
                    intervention.setdefault("duration", "as needed")
                    intervention.setdefault("frequency", "as needed")
                    intervention.setdefault("expected_outcome", "Improved mental wellbeing")
                    intervention.setdefault("rationale", "Based on assessment")
                return interventions
                
            if not json_match:
                logger.error("No JSON array found in response")
                logger.error(f"Full response text: {response_text}")
                return generate_default_interventions(assessment)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            logger.error(f"Response content: {response_text}")
            return generate_default_interventions(assessment)
            
    except Exception as e:
        logger.error(f"Error generating interventions: {str(e)}")
        logger.error(f"Full error details: {e.__dict__}")
        return generate_default_interventions(assessment)

def generate_default_interventions(assessment: Dict) -> List[Dict]:
    """Generate default interventions when Together API fails."""
    logger.warning("Using default interventions due to API failure")
    return [
        {
            "type": "emergency_support",
            "description": "Please contact a mental health professional immediately for support",
            "duration": "immediate",
            "frequency": "as needed",
            "expected_outcome": "Professional support and guidance",
            "rationale": "Based on the assessment indicating need for professional support"
        }
    ] 