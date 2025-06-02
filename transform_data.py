import numpy as np
import glob
import os

import features

def main():
    # Load all of the data files from the data folder
    landmarksFiles = glob.glob(os.path.join("data/", "*_landmark.npy"))
    distancesFiles = glob.glob(os.path.join("data/", "*_distance.npy"))

    landmarks = [np.load(file) for file in landmarksFiles]
    distances = [np.load(file) for file in distancesFiles]
    
    print(f"Loaded {len(landmarks)} landmark files")
    print(f"Loaded {len(distances)} distances files")
    
    if len(landmarks) < 1 or len(distances) < 1:
        print("No Files found in the data folder, validate files and folders and try again")
        raise Exception("files not found!")
    elif len(landmarks) != len(distances):
        raise Exception("missing data files!")
    
    # make the processed directory to save files
    try: 
        os.mkdir("data/processed")
        print("processed folder created")
    except FileExistsError:
        print("processed folder exist, write to folder")
    except Exception as e:
        raise Exception(f"error creating the data/processed folder: {e}")
           
    print("Starting transformations!")
    
    # Process data and save to arrays
    landmarksProcessed = [points_to_features(signal, False) for signal in landmarks ]
    binaryProcessed = [process_to_binary_signal(signal, False) for signal in distances]
    
    # Concatenate the data into input and output data for the NN model
    input = np.concatenate(landmarksProcessed)
    output = np.concatenate(binaryProcessed)
    
    print(f"Input Shape: {input.shape}")
    print(f"Output Shape: {output.shape}")
    
    # Save the data to files
    np.save("data/processed/input.npy", input)
    np.save("data/processed/output.npy", output)
    

# -- Signal Processing Functions --

# Transform the distance from reference frame signal into time-shifted early asserted binary signal
# If verbose is True, the function will print transformation details
def process_to_binary_signal(distances, verbose):
    middle_distance = (np.mean(distances) // 2) # Threshold for the binary decision

    # Map the decision function on the distances array
    # distances values to binary values
    roundBinary = np.vectorize(lambda t: 1 if (t < middle_distance) else 0)
    binary_report = roundBinary(distances)
    
    if verbose:
        print(f"Middle Distance Value = {middle_distance}")
        c = count_rep(binary_report)
        print(f"Number Of Self-Reports: {c}")
    
    # early assertion 13 seconds before self-report
    binary_report = add_flag(binary_report)
    
    # To time-shift the signal 8 seconds backward (8 seconds * 30 fps = 240 frames)
    # first 240 frames are removed and 240 frames are concatenated to the end of the signal      
    binary_report = np.concatenate((np.split(binary_report, [240])[1], np.zeros(240)))
    
    return binary_report
  
# Transform the (n,68,2) landmark points tensor into (n,14) features matrix (n is the number of frames / length of the array)
# If verbose is true, the function will print n
# Features in the feature_matrix by index:
# [IMAR, LEAR, REAR, LEEAR, REEAR, NXP, NYP, OMAR, FXP, FYP, LFRR, RFRR, MNAR, MCAR]
def points_to_features(points, verbose):
    
    # To transform the (n, 68, 2) landmark tensor into the (n, 14) features matrix 
    # (where n is the number of frames) the features module is needed 
    n = points.shape[0]
    
    if verbose:
        print(f"n = {n} frames")

    # Initialize the empty matrix of shape (n,14)
    feature_matrix = np.ndarray((n, 14))

    # Could take few a minutes for big arrays
    for idx, landmark_matrix in enumerate(points):
        feature_matrix[idx] = features.convert_landmark_points_to_features(landmark_matrix)
        
    return feature_matrix
      

# -- Helper Functions -- 
    
# Find the number of reports (210 frames per report / 7 seconds)
def count_rep(arr):
    i = 0
    count = 0
    while i < len(arr):
        if arr[i] == 1:
            i += 240 # skip 8 seconds 
            count += 1
        else:
            i += 1
    return count


# To add 13 seconds of report (for a total of 20 seconds of positive report) before the self-report, (13 seconds * 30 fps = 390 frames)
# 390 frames before the signal should be marked
def add_flag(binary_report):
    i = 0
    while i < len(binary_report):
        # Flag 390 frames back
        if binary_report[i] == 1:
            if i-390 >= 0:
                binary_report[i-390: i] = 1
        # go to next frame
        i += 1
    return binary_report

if __name__ == "__main__":
    main()