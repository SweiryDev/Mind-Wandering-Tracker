import cv2
import dlib
import numpy as np

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

videoName = "../AmitS.mkv"

# Set video to grab and decode
cap = cv2.VideoCapture(videoName)

# Set the video to start at the first frame
currentFrame = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)

# Get the number of frames in the video
videoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Number Of Frames In The Video: {videoFrames}")

# Initialize the landmark array (array of landmark matrices)
landmarks_tensor = []

print("Starting Landmark extraction PLEASE WAIT!")

# Iterating frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if currentFrame % 100 == 0:
        print(f"Progress: {round(100 * (currentFrame / videoFrames), 2)}%", end="\r")
        
    # Convert RGB frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Make sure to catch a face in the frame
    if not faces:
        continue
    face = faces[0]
    
    # Get facial landmarks and append to array
    landmarks = predictor(gray, face)
    landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
    landmarks_tensor.append(landmarks_points)
    

landmarks_tensor = np.array(landmarks_tensor, dtype=int)
np.save("data/landmark.npy", landmarks_tensor)
print(f"Landmark Tensor Shape: {landmarks_tensor.shape}, Saved to data/landmark.npy")

cap.release()
cv2.destroyAllWindows()
