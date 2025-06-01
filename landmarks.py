import cv2
import dlib
import numpy as np

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

videoName = "../AviC.mkv"
cap = cv2.VideoCapture(videoName)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        i = 1

        for landmark in landmarks_points:
            cv2.circle(frame, tuple(landmark), 3, (0, 255, 0), -1)
            cv2.putText(frame, f"{i}", tuple(landmark), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0))
            i += 1
        
        cv2.putText(frame, f"{round(cap.get(cv2.CAP_PROP_POS_FRAMES))}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))
    
    cv2.imshow("Display Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
