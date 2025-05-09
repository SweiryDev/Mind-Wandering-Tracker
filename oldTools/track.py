import cv2
import numpy as np
import os
import time

def main():
    # Find cascade files - try several common locations
    cascade_paths = [
        '/usr/share/opencv4/haarcascades/',  # Ubuntu system install location
        '/usr/local/share/opencv4/haarcascades/',
        '/usr/share/opencv/haarcascades/',
        '/usr/local/share/opencv/haarcascades/'
    ]
    
    face_cascade_path = None
    eye_cascade_path = None
    
    for path in cascade_paths:
        if os.path.exists(path + 'haarcascade_frontalface_default.xml'):
            face_cascade_path = path + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = path + 'haarcascade_eye.xml'
            break
    
    if face_cascade_path is None:
        print("Error: Could not find Haar cascade XML files.")
        print("Please install OpenCV data files with:")
        print("sudo apt install opencv-data")
        return
    
    # Load pre-trained classifiers
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    print(f"Using cascade files from: {os.path.dirname(face_cascade_path)}")
    
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Set desired FPS parameters
    target_fps = 5  # Process only 5 frames per second
    frame_interval = 1.0 / target_fps  # Time between frames
    
    # Variables for frame rate control
    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Time control for frame rate
        current_time = time.time()
        elapsed = current_time - prev_frame_time
        
        # Only process frames at the target rate
        if elapsed > frame_interval:
            prev_frame_time = current_time
            
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Region of interest for face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangle around eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Calculate eye center
                    eye_center_x = x + ex + ew // 2
                    eye_center_y = y + ey + eh // 2
                    
                    # Draw circle at eye center
                    cv2.circle(frame, (eye_center_x, eye_center_y), 3, (0, 0, 255), -1)
                    
                    # Add text with eye coordinates
                    cv2.putText(frame, f"Eye: ({eye_center_x}, {eye_center_y})", 
                               (eye_center_x - 60, eye_center_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Calculate and display current FPS
            frame_count += 1
            new_frame_time = time.time()
            total_elapsed = new_frame_time - start_time
            
            if total_elapsed > 0:
                current_fps = frame_count / total_elapsed
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Eye Tracking', frame)
        
        # Always check for key press (with minimal delay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Display final statistics
    if frame_count > 0 and (time.time() - start_time) > 0:
        average_fps = frame_count / (time.time() - start_time)
        print(f"Average processing rate: {average_fps:.2f} FPS")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()