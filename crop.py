import cv2
import dlib
import numpy as np
from typing import NamedTuple

videoName = "../AmitS.mkv"
i = 815 # Start Emoji

selectiveArea = [1248, 19, 14, 19]
cropRec = []


class ReferenceFrame(NamedTuple):
    roiRec: tuple[int, int, int, int]
    refPixel: tuple[int, int, int]

def main(i, videoName):
    # Declare global var
    global cropRec

    while True:
        # Get video from file and set frame
        cap = cv2.VideoCapture(videoName)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Print current frame and read frame
        print(f"Load Frame: {i}")
        ret, frame = cap.read()
        if not ret:
            print("Error")
            return

        # Check for declared roi     
        if len(cropRec) < 1:
            cropRec = selectCrop(frame)
        
        # Crop frame
        croppedFrame = frame[cropRec[1] : (cropRec[1] + cropRec[3]), cropRec[0]: (cropRec[0] + cropRec[2])]

        # Calculate average pixel values from the cropped frame
        cframe_arr = croppedFrame.flatten().reshape((-1, 3))
        avg_pixel = np.sum(cframe_arr, axis=0) // (cframe_arr.shape[0] * cframe_arr.shape[1])
        print(avg_pixel)

        # Show frame
        cv2.imshow("FRAME", croppedFrame)

        # Get key press and move frames 
        key = cv2.waitKey(0)
        if key == 83 or key == ord('d'): # Right arrow / d key
            i += 1
        elif key == 82 or key == ord('w'): # Up arrow / w key
            i += 10
        elif key == 84 or key == ord('s'): # Down arrow / s key
            i -= 10
        elif key == 81 or key == ord('a') : # Left arrow / a key
            i -= 1
        elif key == 49 or key == ord('1'): # '1' key
            print(f"Chosen Average Pixel: {avg_pixel}")
            cv2.destroyAllWindows()
            return ReferenceFrame(cropRec, avg_pixel)
        else:
            cap.release()
            cv2.destroyAllWindows()
            return
        
def selectCrop(frame):
    roi = cv2.selectROI("SelectROI", frame, False)
    cv2.destroyWindow("SelectROI")
    return np.array(roi)
