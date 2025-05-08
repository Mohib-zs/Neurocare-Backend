from fastapi import APIRouter, File, UploadFile, WebSocket, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import numpy as np
import cv2
from datetime import datetime
import base64

from .database import get_db
from .realtime_utils import RealtimeEmotionAnalyzer
from . import crud, schemas

router = APIRouter()

# Store analyzers for each user
user_analyzers: Dict[int, RealtimeEmotionAnalyzer] = {}

def get_analyzer(user_id: int) -> RealtimeEmotionAnalyzer:
    if user_id not in user_analyzers:
        user_analyzers[user_id] = RealtimeEmotionAnalyzer()
    return user_analyzers[user_id]

@router.post("/emotion/analyze/image")
async def analyze_single_image(
    file: UploadFile = File(...),
    user_id: int = None,
    db: Session = Depends(get_db)
):
    """
    Analyze emotions in a single uploaded image.
    Following the UPLOAD_PROMPT structure for consistent response format.
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get or create analyzer
        analyzer = get_analyzer(user_id) if user_id else RealtimeEmotionAnalyzer()
        
        # Process image
        result = analyzer.process_frame(image)
        
        if result["status"] != "success":
            raise HTTPException(status_code=422, detail="Failed to process image")
            
        # Save analysis if user is authenticated
        if user_id and db:
            analysis = crud.create_emotion_analysis(
                db=db,
                user_id=user_id,
                emotions=result["emotion_scores"],
                dominant_emotion=result["dominant_emotion"]
            )
            
            # Generate intervention if needed
            if analyzer.should_trigger_intervention():
                intervention = analyzer.generate_realtime_intervention()
                if intervention:
                    analysis.intervention = intervention
                    db.commit()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.post("/emotion/analyze/base64")
async def analyze_base64_image(
    data: schemas.Base64Image,
    user_id: int = None,
    db: Session = Depends(get_db)
):
    """
    Analyze emotions in a base64 encoded image.
    Useful for web applications sending canvas/webcam data.
    """
    try:
        # Get or create analyzer
        analyzer = get_analyzer(user_id) if user_id else RealtimeEmotionAnalyzer()
        
        # Process base64 image
        result = analyzer.process_frame(data.image, is_base64=True)
        
        if result["status"] != "success":
            raise HTTPException(status_code=422, detail="Failed to process image")
            
        # Save analysis if user is authenticated
        if user_id and db:
            analysis = crud.create_emotion_analysis(
                db=db,
                user_id=user_id,
                emotions=result["emotion_scores"],
                dominant_emotion=result["dominant_emotion"]
            )
            
            # Generate intervention if needed
            if analyzer.should_trigger_intervention():
                intervention = analyzer.generate_realtime_intervention()
                if intervention:
                    analysis.intervention = intervention
                    db.commit()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.websocket("/ws/emotion/realtime/{user_id}")
async def realtime_emotion_analysis(
    websocket: WebSocket,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time emotion analysis.
    Follows the REALTIME_PROMPT structure for streaming analysis.
    """
    try:
        await websocket.accept()
        analyzer = get_analyzer(user_id)
        
        # Create a new session
        session = crud.create_emotion_session(
            db=db,
            user_id=user_id,
            session_type=schemas.SessionType.REALTIME
        )
        
        try:
            while True:
                # Receive frame data
                data = await websocket.receive_json()
                
                if "frame" not in data:
                    await websocket.send_json({
                        "status": "error",
                        "detail": "No frame data received"
                    })
                    continue
                
                # Process frame
                result = analyzer.process_frame(data["frame"], is_base64=True)
                
                # Save frame analysis
                if result["status"] == "success":
                    frame_analysis = crud.create_realtime_frame(
                        db=db,
                        session_id=session.id,
                        emotions=result["emotion_scores"],
                        dominant_emotion=result["dominant_emotion"],
                        confidence_score=result["confidence_score"]
                    )
                    
                    # Check for intervention need
                    if analyzer.should_trigger_intervention():
                        intervention = analyzer.generate_realtime_intervention()
                        if intervention:
                            result["intervention"] = intervention
                            
                            # Save intervention
                            crud.add_session_intervention(
                                db=db,
                                session_id=session.id,
                                intervention=intervention
                            )
                
                # Send analysis results
                await websocket.send_json(result)
                
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            await websocket.close()
            
        finally:
            # Update session end time and generate summary
            session.end_time = datetime.now()
            summary = analyzer.generate_session_summary()
            session.summary = summary
            db.commit()
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        
    finally:
        if user_id in user_analyzers:
            user_analyzers[user_id].cleanup()

@router.get("/emotion/session/{session_id}/summary")
async def get_session_summary(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get summary for a completed emotion analysis session."""
    session = crud.get_emotion_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return JSONResponse(content={
        "session_id": session.id,
        "start_time": session.start_time.isoformat(),
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "summary": session.summary,
        "interventions": session.interventions
    }) 