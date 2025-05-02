import cv2
import numpy as np
import os
import time

def main():
    # Load pre-trained classifiers
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)

    while True:
        # Get frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Color to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        screenSize = len(gray)

        # Detect faces and eyes
        # faces = face_cascade.detectMultiScale(gray)
        eyes = eye_cascade.detectMultiScale(gray)

        for (ex, ey, ew, eh) in eyes:    
            eye_x = ex + ew // 2
            eye_y = ey + eh // 2

            cv2.circle(gray, (eye_x,eye_y), 3,(255,0,0), -1)
            cv2.putText(gray, f"EYE H: {eh}"
                        , (eye_x - 10, eye_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1)


        # Show Frame
        cv2.imshow('Frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

