import cv2
import dlib
import numpy as np
import crop

videoName = "../AmitS.mkv"
lastFrame = 0

# Offset frame to start the manual frame selection (refrence roi and average pixel)
startFrameRef = 815


def main():
    global ref
    
    # Get the reference rectangle of the frame and reference pixel (detect self-report)
    ref = crop.main(startFrameRef)
    if not ref: raise Exception("No reference frame and average pixel!")

    # Load face detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

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
    # Initialize the pixel reference distance array( array of frame average pixel distance from the reference frame - self report )
    pixel_distance_arr = []

    print("Starting Landmark extraction PLEASE WAIT!")

    # Iterate frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if currentFrame % 100 == 0:
            print(f"Progress: {round(100 * (currentFrame / videoFrames), 2)}%", end="\r")
            
        # Convert RGB frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get landmarks from grayscale frame (Skip iteration if no face detected!)
        landmarks_points = GetLandmarkPoints(detector, predictor, gray)
        if landmarks_points is None: continue
        
        # append points to tensor
        landmarks_tensor.append(landmarks_points)
        
        # Get the distance from reference average frame to identify self report
        pixel_distance = GetPixelDistanceFromRef(frame)
        
        # append the current pixel distance to the array
        pixel_distance_arr.append(pixel_distance)
        
        # Limit the number of frames to process
        if lastFrame > 0 and lastFrame == currentFrame: break
        
    # Save landmarks tensor to file
    landmarks_tensor = np.array(landmarks_tensor, dtype=int)
    np.save("data/landmark.npy", landmarks_tensor)
    print(f"Landmark Tensor Shape: {landmarks_tensor.shape}, Saved to data/landmark.npy")
    
    # Save pixel distance array to file
    pixel_distance_arr = np.array(pixel_distance_arr, dtype=int)
    np.save("data/distance.npy", pixel_distance_arr)
    print(f"Pixel Distance Shape: {pixel_distance_arr.shape}, Saved to data/distance.npy")

    # Release the video file
    cap.release()

def GetLandmarkPoints(detector, predictor, grayFrame):
        faces = detector(grayFrame)
        
        # Make sure to catch a face in the frame
        if not faces:
            return 
        face = faces[0]
        
        # Get facial landmarks and append to array
        landmarks = predictor(grayFrame, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        return landmarks_points

def GetPixelDistanceFromRef(frame):
        # Crop frame
        cropped_frame = frame[ref.roiRec[1] : (ref.roiRec[1] + ref.roiRec[3]), ref.roiRec[0]: (ref.roiRec[0] + ref.roiRec[2])]

        # Calculate average pixel values from the cropped frame
        cframe_arr = cropped_frame.flatten().reshape((-1, 3))
        avg_pixel = np.sum(cframe_arr, axis=0) // (cframe_arr.shape[0] * cframe_arr.shape[1])
        
        # Calculate the pixel distance of current average pixel to reference pixel
        pixel_distance = np.array(avg_pixel, int) - np.array(ref.refPixel, int)
        pixel_distance = round(np.linalg.norm(pixel_distance))
        return pixel_distance

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()

    