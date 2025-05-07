import cv2
import dlib
import numpy as np

videoName = "../AmitS.mkv"
i = 808 # Start Emoji

selectiveArea = [1248, 19, 14, 19]
cropRec = selectiveArea

def main():
    # Declare global var
    global i, cropRec

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
        if key == 83: # Right arrow key
            i += 1
        elif key == 82: # Up arrow key
            i += 10
        elif key == 84: # Down arrow key
            i -= 10
        elif key == 81: # Left arrow key
            i -= 1
        elif key == 49: # '1' key
            print(f"Chosen Average Pixel: {avg_pixel}")
        else:
            cap.release()
            return
        

def selectCrop(frame):
    roi = cv2.selectROI("SelectROI", frame, False)
    cv2.destroyWindow("SelectROI")
    return np.array(roi)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()    


