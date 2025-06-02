import cv2
import dlib
import numpy as np

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Function to get eye aspect ratio (EAR)
def eye_aspect_ratio(eye_points):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Start video capture (0 - Camera / 1 - Virtual Camera)
cap = cv2.VideoCapture("../AmitS.mkv")

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
        
        # Get eye landmarks (36-41 for left eye, 42-47 for right eye)
        left_eye = landmarks_points[36:42]
        right_eye = landmarks_points[42:48]
        
        # Calculate the center of each eye
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)
        
        # Draw circles at eye centers
        cv2.circle(frame, tuple(left_eye_center), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_eye_center), 3, (0, 255, 0), -1)
        
        # Draw eye contours
        cv2.polylines(frame, [left_eye], True, (255, 0, 0), 1)
        cv2.polylines(frame, [right_eye], True, (255, 0, 0), 1)
        
        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Display EAR value
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()