import cv2

def capture_test_image():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press SPACE to capture image or ESC to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.imshow('Capture Test Image', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If ESC pressed, exit
        if key == 27:  # ESC key
            break
        # If SPACE pressed, save image
        elif key == 32:  # SPACE key
            cv2.imwrite('test_image.jpg', frame)
            print("Image saved as 'test_image.jpg'")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_test_image() 