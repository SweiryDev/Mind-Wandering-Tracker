# Mind Wandering Tracker

This repo contains the source code of the mind wandering tracker project for "Image Processing Lab" course

The experiment was conducted on using Google Meet to capture the participant reading and emoji-based self-report in the same video.

## Installation

Clone this repo using git

`git clone https://github.com/SweiryDev/Mind-Wandering-Tracker.git`

### Requirements

- VSCode (tested) or another Jupyter env (for python notebook)
- Python 3.10 (tested on 3.10.9)
- CMake
- pip modules: numpy, opencv-python, dlib, matplotlib, ipykernal, tensorflow, scikit-learn

check the Tensorflow docs for the optimal installation for your OS and GPU.

this repo is required with data folder including read/write access.

## Usage

The repo contains few script to extract data from participant videos, it doesn't contain any videos or data.

30 fps video (tested with mkv files) or webcam is needed for 68 landmark points and self-report signal.

1. Conduct the experiment on a video chat platform with an on-screen emoji message appearance, let the participant read the experiment text and send clapping-hands emoji when they detect mind wandering.

2. Test the landmark points model (shape_predictor_68_face_landmarks.dat) use the landmarks.py script, and modify the file name like the following lines:

``` 
videoName = 0 # for webcam live usage 

videoName = "path_to_video" # to load and play video file
``` 

3. Extract the 68 landmark points and the self-report signal using the extractVideo.py script.
Once you run it you have to select the region of interest (ROI) where the emoji fully appear (press enter to confirm), the next window is the reference frame selection window (press 'd'/'a' to move 1 frame forward/backward and the key '1' to confirm or 'c' to exit).
The script will output 2 data files into the data folder.

4. Validate data integrity using the data_test.ipynb python notebook, data folder with the extracted .npy files is needed.
modify the Name variable with the name of the video (mkv video, it's possible to change the path to video in the np.load function).
Go over the code blocks to validate the data and confirm the number of self-reports.

5. Transform the valid landmark points data and binary self-report signal using the **transformData.py** script.

...