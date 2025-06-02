# Mind Wandering Tracker

This repo contains the source code of the mind wandering tracker project for "Image Processing Lab" course

The experiment was conducted on using Google Meet to capture the participant reading and emoji-based self-report in the same video.

## Installation

Clone this repo using git

`git clone https://github.com/SweiryDev/Mind-Wandering-Tracker.git`

### Requirements

- VSCode (tested) or another Jupyter enviorment (for python notebook)
- Python 3.10 (tested on 3.10.9)
- CMake
- pip modules: numpy, opencv-python, dlib, matplotlib, ipykernal, tensorflow, scikit-learn

check the Tensorflow docs for the optimal installation for your OS and GPU, using WSL for GPU acceleration is recommended.
for WSL and linux use pyenv.

this repo is required with data folder including read/write access.


## Usage

The repo contains few script to extract data from participant videos, it doesn't contain any videos or data.

30 fps video (tested with mkv files) or webcam is needed for 68 landmark points and self-report signal.

Run the scripts mentioned in the guide using py (in windows terminal):

`py -3.10 .\script_name.py # replace script_name with the script you want to run`

1. Conduct the experiment on a video chat platform with an on-screen emoji message pop-up, let the participant read the experiment text and ask them to send "clapping hands" emoji when they detect mind wandering (self-report).

2. Test the landmark points model (shape_predictor_68_face_landmarks.dat) use the landmarks.py script, and modify the file name like the following lines:

``` 
videoName = 0 # for webcam live usage 

videoName = "path_to_video" # to load and play video file
``` 

3. Extract the 68 landmark points and the self-report signal using the extractVideo.py script.
Change the Name variable in the script for the output file name to contain the participant name, change the StartFrameRef variable to the frame when the emoji appear in the experiment video.
Once you run it you have to select the region of interest (ROI) where the emoji fully appear (press enter to confirm), the next window is the reference frame selection window (press 'd'/'a' to move 1 frame forward/backward and the key '1' to confirm or 'c' to exit).
The script will output 2 data files into the data folder.

4. Validate data integrity using the data_test.ipynb python notebook (the distance signal should fluctuate between two values 0 and max distance with a small error of about ~10 units, if the error is bigger consider a more focused ROI in step 3.), data folder with the extracted .npy files is needed.
modify the Name variable with the name of the video (mkv video, it's possible to change the path to video in the np.load function).
Go over the code blocks to validate the data and confirm the number of self-reports.

5. Transform the valid landmark points data and self-report distance signal using the transform_data.py script, the main function in the script process the data into the input.npy and output.npy files (of shape (n, 14) and (n,) respectively) and saves it to data/process folder.


Training and testing neural network coming soon...
