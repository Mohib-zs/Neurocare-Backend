import asyncio
import websockets
import requests
import json
import base64
import cv2
import os
from datetime import datetime

# API endpoints
BASE_URL = "http://localhost:8000"
SINGLE_IMAGE_URL = f"{BASE_URL}/emotion/analyze/image"
BASE64_URL = f"{BASE_URL}/emotion/analyze/base64"
WEBSOCKET_URL = f"ws://localhost:8000/ws/emotion/realtime"

async def test_single_image_upload():
    """Test uploading a single image for emotion analysis"""
    print("\n=== Testing Single Image Upload ===")
    
    # Replace with path to your test image
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please place a test image at {image_path}")
        return
        
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': ('test_image.jpg', image_file, 'image/jpeg')}
            params = {'user_id': 1}  # Replace with actual user ID if needed
            
            response = requests.post(SINGLE_IMAGE_URL, files=files, params=params)
            
            print(f"Status Code: {response.status_code}")
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print(f"Error testing single image upload: {str(e)}")

async def test_base64_image():
    """Test base64 image analysis"""
    print("\n=== Testing Base64 Image Analysis ===")
    
    # Replace with path to your test image
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please place a test image at {image_path}")
        return
        
    try:
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        payload = {
            "image": f"data:image/jpeg;base64,{image_data}",
            "user_id": 1  # Replace with actual user ID if needed
        }
        
        response = requests.post(BASE64_URL, json=payload)
        
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Error testing base64 image: {str(e)}")

async def test_realtime_analysis():
    """Test real-time emotion analysis using webcam"""
    print("\n=== Testing Real-time Analysis ===")
    print("Press 'q' to stop the webcam test")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    try:
        # Connect to WebSocket
        user_id = 1  # Replace with actual user ID if needed
        async with websockets.connect(f"{WEBSOCKET_URL}/{user_id}") as websocket:
            print("WebSocket connected")
            
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Display frame
                cv2.imshow('Webcam Test', frame)
                
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame
                await websocket.send(json.dumps({
                    "frame": f"data:image/jpeg;base64,{base64_frame}"
                }))
                
                # Receive analysis
                response = await websocket.recv()
                result = json.loads(response)
                
                # Print analysis results
                print(f"\rDominant Emotion: {result.get('dominant_emotion', 'unknown')} "
                      f"(Confidence: {result.get('confidence_score', 0):.2f})", end='')
                
                # Check for intervention
                if 'intervention' in result:
                    print(f"\nIntervention: {result['intervention']}")
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"\nError in real-time analysis: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def test_session_summary(session_id: int = 1):
    """Test getting session summary"""
    print(f"\n=== Testing Session Summary (ID: {session_id}) ===")
    
    try:
        response = requests.get(f"{BASE_URL}/emotion/session/{session_id}/summary")
        
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
    except Exception as e:
        print(f"Error getting session summary: {str(e)}")

async def main():
    """Run all tests"""
    # Test single image upload
    await test_single_image_upload()
    
    # Test base64 image
    await test_base64_image()
    
    # Test real-time analysis
    await test_realtime_analysis()
    
    # Test session summary
    await test_session_summary()

if __name__ == "__main__":
    asyncio.run(main()) 