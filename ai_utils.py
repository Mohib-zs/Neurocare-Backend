from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def analyze_mental_health_trends(trends: Dict[str, List[float]]) -> Dict[str, Any]:
    """Analyze trends in mental health metrics and generate insights."""
    try:
        insights = {
            "overall_trend": {},
            "key_improvements": [],
            "areas_of_concern": [],
            "recommendations": []
        }
        
        for metric, values in trends.items():
            if not values:
                continue
                
            # Calculate basic statistics
            mean = np.mean(values)
            std = np.std(values)
            trend = np.polyfit(range(len(values)), values, 1)[0]
            
            # Store trend information
            insights["overall_trend"][metric] = {
                "mean": float(mean),
                "std": float(std),
                "trend": float(trend),
                "improvement": trend < 0,  # Negative trend indicates improvement
                "stability": std < 0.5  # Low standard deviation indicates stability
            }
            
            # Identify significant changes
            if abs(trend) > 0.1:  # Significant trend
                if trend < 0:
                    insights["key_improvements"].append({
                        "metric": metric,
                        "improvement": float(abs(trend)),
                        "description": f"Significant improvement in {metric}"
                    })
                else:
                    insights["areas_of_concern"].append({
                        "metric": metric,
                        "deterioration": float(trend),
                        "description": f"Concerning trend in {metric}"
                    })
        
        # Generate recommendations based on insights
        insights["recommendations"] = generate_recommendations(insights)
        
        return insights
        
    except Exception as e:
        logger.error(f"Error analyzing mental health trends: {str(e)}")
        return None

def generate_recommendations(insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate personalized recommendations based on insights."""
    recommendations = []
    
    # Analyze areas of concern
    for concern in insights["areas_of_concern"]:
        metric = concern["metric"]
        if metric == "depression":
            recommendations.append({
                "type": "therapeutic",
                "priority": "high",
                "description": "Consider scheduling a therapy session to address depressive symptoms",
                "action_items": [
                    "Schedule regular therapy sessions",
                    "Practice daily mood tracking",
                    "Engage in physical activity"
                ]
            })
        elif metric == "anxiety":
            recommendations.append({
                "type": "mindfulness",
                "priority": "medium",
                "description": "Implement daily mindfulness practices to manage anxiety",
                "action_items": [
                    "Start with 5-minute meditation sessions",
                    "Practice deep breathing exercises",
                    "Use grounding techniques when anxious"
                ]
            })
        elif metric == "sleep_quality":
            recommendations.append({
                "type": "lifestyle",
                "priority": "medium",
                "description": "Improve sleep hygiene and establish consistent sleep patterns",
                "action_items": [
                    "Maintain regular sleep schedule",
                    "Create a relaxing bedtime routine",
                    "Limit screen time before bed"
                ]
            })
    
    # Add general wellness recommendations
    recommendations.append({
        "type": "general",
        "priority": "low",
        "description": "Maintain overall mental wellness",
        "action_items": [
            "Stay connected with support network",
            "Practice self-care regularly",
            "Keep a gratitude journal"
        ]
    })
    
    return recommendations

def generate_intervention_feedback(
    progress_metrics: Dict[str, Any],
    intervention_type: str
) -> Dict[str, Any]:
    """Generate AI feedback for intervention progress."""
    try:
        feedback = {
            "overall_progress": {},
            "strengths": [],
            "challenges": [],
            "next_steps": []
        }
        
        # Analyze adherence
        adherence = progress_metrics.get("adherence_score", 0)
        feedback["overall_progress"]["adherence"] = {
            "score": adherence,
            "status": "good" if adherence >= 0.7 else "needs_improvement"
        }
        
        # Analyze effectiveness
        effectiveness = progress_metrics.get("effectiveness_score", 0)
        feedback["overall_progress"]["effectiveness"] = {
            "score": effectiveness,
            "status": "good" if effectiveness >= 0.7 else "needs_improvement"
        }
        
        # Generate specific feedback based on intervention type
        if intervention_type == "therapeutic":
            feedback["strengths"].append("Regular engagement with therapeutic activities")
            feedback["challenges"].append("Maintaining consistent practice between sessions")
            feedback["next_steps"].append({
                "action": "Schedule follow-up session",
                "timeline": "within 1 week",
                "priority": "high"
            })
        elif intervention_type == "mindfulness":
            feedback["strengths"].append("Developing awareness of present moment")
            feedback["challenges"].append("Finding time for daily practice")
            feedback["next_steps"].append({
                "action": "Increase practice duration gradually",
                "timeline": "over 2 weeks",
                "priority": "medium"
            })
        elif intervention_type == "lifestyle":
            feedback["strengths"].append("Implementing healthy habits")
            feedback["challenges"].append("Maintaining consistency in new routines")
            feedback["next_steps"].append({
                "action": "Set specific, achievable goals",
                "timeline": "weekly",
                "priority": "medium"
            })
        
        return feedback
        
    except Exception as e:
        logger.error(f"Error generating intervention feedback: {str(e)}")
        return None

def predict_outcomes(
    current_metrics: Dict[str, float],
    intervention_plan: Dict[str, Any]
) -> Dict[str, Any]:
    """Predict potential outcomes based on current metrics and intervention plan."""
    try:
        predictions = {
            "short_term": {},
            "long_term": {},
            "confidence_level": {},
            "risk_factors": []
        }
        
        # Analyze each metric
        for metric, value in current_metrics.items():
            # Short-term predictions (1-2 weeks)
            predictions["short_term"][metric] = {
                "expected_change": calculate_expected_change(value, intervention_plan),
                "confidence": 0.7  # Base confidence level
            }
            
            # Long-term predictions (3-6 months)
            predictions["long_term"][metric] = {
                "expected_change": calculate_expected_change(value, intervention_plan, long_term=True),
                "confidence": 0.5  # Lower confidence for long-term predictions
            }
        
        # Identify potential risk factors
        predictions["risk_factors"] = identify_risk_factors(current_metrics, intervention_plan)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error predicting outcomes: {str(e)}")
        return None

def calculate_expected_change(
    current_value: float,
    intervention_plan: Dict[str, Any],
    long_term: bool = False
) -> float:
    """Calculate expected change in a metric based on intervention plan."""
    # Base improvement rate
    improvement_rate = 0.1  # 10% improvement per week
    
    # Adjust for intervention type
    if intervention_plan.get("type") == "therapeutic":
        improvement_rate *= 1.2
    elif intervention_plan.get("type") == "mindfulness":
        improvement_rate *= 1.1
    elif intervention_plan.get("type") == "lifestyle":
        improvement_rate *= 0.9
    
    # Adjust for long-term predictions
    if long_term:
        improvement_rate *= 0.7  # Diminishing returns over time
    
    return current_value * (1 - improvement_rate)  # Negative change indicates improvement

def identify_risk_factors(
    current_metrics: Dict[str, float],
    intervention_plan: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify potential risk factors based on current metrics and intervention plan."""
    risk_factors = []
    
    # Check for high risk levels
    if current_metrics.get("depression_score", 0) > 0.7:
        risk_factors.append({
            "type": "depression",
            "level": "high",
            "description": "Elevated depression symptoms",
            "recommendation": "Consider immediate professional support"
        })
    
    if current_metrics.get("anxiety_score", 0) > 0.7:
        risk_factors.append({
            "type": "anxiety",
            "level": "high",
            "description": "Elevated anxiety symptoms",
            "recommendation": "Implement immediate stress management techniques"
        })
    
    if current_metrics.get("sleep_quality_score", 0) < 0.3:
        risk_factors.append({
            "type": "sleep",
            "level": "medium",
            "description": "Poor sleep quality",
            "recommendation": "Review and improve sleep hygiene practices"
        })
    
    return risk_factors 