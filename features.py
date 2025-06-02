import cv2
import dlib
import numpy as np

# Function maps the points to features and returns an array of the features as described below
# [IMAR, LEAR, REAR, LEEAR, REEAR, NXP, NYP, OMAR, FXP, FYP, LFRR, RFRR, MNAR, MCAR]
def convert_landmark_points_to_features(points):
    # -- IMAR -- 
    inner_mouth_points = points[60:68] # P61 - P68
    imar = inner_mouth_aspect_ratio(inner_mouth_points)
    
    # -- EAR --
    left_eye_points = points[36:42] # P37 - P42
    right_eye_points = points[42:48] # P43 - P48
    
    lear = eye_aspect_ratio(left_eye_points)
    rear = eye_aspect_ratio(right_eye_points)
    
    # -- EEAR --
    left_eyebrow_points = points[17:22] # P18 to P22
    right_eyebrow_points = points[22:27] # P23 to P27
    
    leear = eyebrow_eye_aspect_ratio(left_eyebrow_points, left_eye_points)
    reear = eyebrow_eye_aspect_ratio(right_eyebrow_points, right_eye_points)
    
    # -- NP -- 
    nose_points = points[27:36] # P28 to P36 
    nose_center = nose_point(nose_points) 
    (nxp, nyp) = nose_center
    
    # -- OMAR -- 
    outer_mouth_points = points[48:60] # P49 - P60        
    omar = outer_mouth_aspect_ratio(outer_mouth_points)
    
    # -- FP --
    face_points = points # P1 - P68
    (fxp, fyp) = face_point(face_points) 
    
    # -- FRR --
    left_cheek_points = points[0:5] # P1 - P5
    right_cheek_points = points[12 : 17] # P13 - P17
    
    (lfrr, rfrr) = face_rotation_ratio(nose_points, left_cheek_points, right_cheek_points)
    
    # -- MNAR -- 
    upper_lip_points = points[48:55] # P49 - P55
    
    mnar = mouth_nose_aspect_ratio(nose_points, upper_lip_points)
    
    # -- MCAR -- 
    lower_lip_points = points[55:60] # P56 - P60
    chin_points = points[7:10] # P8 - P10
    
    mcar = mouth_chin_aspect_ratio(lower_lip_points, chin_points)
    
    return [imar, lear, rear, leear, reear, nxp, nyp, omar, fxp, fyp, lfrr, rfrr, mnar, mcar]

    
    

# Function to get eye aspect ratio (EAR)
def eye_aspect_ratio(eye_points):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye_points[1] - eye_points[5]) # P38 - P42
    B = np.linalg.norm(eye_points[2] - eye_points[4]) # P39 - P41
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_points[0] - eye_points[3]) # P37 - P40
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return round(ear, 2)

# Function to get inner mouth aspect ratio (IMAR)
def inner_mouth_aspect_ratio(mouth_points):
        # Compute three vertical distances
        v1 = np.linalg.norm(mouth_points[2] - mouth_points[6])  # 63–67
        v2 = np.linalg.norm(mouth_points[3] - mouth_points[5])  # 64–66
        v3 = np.linalg.norm(mouth_points[1] - mouth_points[7])  # 62–68
        vertical_avg = (v1 + v2 + v3) / 3.0

        # Compute horizontal mouth width (between 61 and 65)
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[4])  # 61–65

        # Normalized mouth opening ratio (MAR)
        mouth_opening_ratio = vertical_avg / horizontal
        return round(mouth_opening_ratio,2)

# Function to get outer mouth aspect ratio (OMAR) P49 to P60
def outer_mouth_aspect_ratio(mouth_points):
    # vertical distances of outer mouth
    v1 = np.linalg.norm(mouth_points[1] - mouth_points[11])  # P50 - P60
    v2 = np.linalg.norm(mouth_points[2] - mouth_points[10])  # P51 - P59
    v3 = np.linalg.norm(mouth_points[3] - mouth_points[9])  # P52 - P58
    v4 = np.linalg.norm(mouth_points[4] - mouth_points[8])  # P53 - P57
    v5 = np.linalg.norm(mouth_points[5] - mouth_points[7])  # P54 - P56

    # average vertical distance
    vertical_avg = (v1 + v2 + v3 + v4 + v5) / 5.0
    
    # horizontal distance of outer mouth
    horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6]) # P49 - P55
    
    omar = vertical_avg / horizontal
    return round(omar,2)
    

# Function to get eyebrow distance from eye center (EEAR)
def eyebrow_eye_aspect_ratio(eyebrow_points, eye_points):
    # Calculate centers for eye and eyebrow for stable indicators
    eye_center = np.mean(eye_points, axis=0).astype(int)
    eyebrow_center = np.mean(eyebrow_points, axis=0).astype(int)
    
    # Calculate eyebrow horizontal length
    horizontal_distance = np.linalg.norm(eyebrow_points[0] - eyebrow_points[4])
    
    # Vertiacl distance between centers
    vertical_distance = np.linalg.norm(eyebrow_center - eye_center)
    
    eear = vertical_distance / horizontal_distance
    return round(eear, 2)  
    
# Function to get the stable nose tip cooridantes, NXP and NYP
def nose_point(nose_points):
    nose_center = np.mean(nose_points, axis=0).astype(int) # P28 to P36            
    (nxp, nyp) = nose_center
    return (nxp, nyp)

# Function to get the stable face center coordiantes, FXP and FYP (all of the landmark points)
def face_point(face_points):
    face_center = np.mean(face_points, axis=0).astype(int)
    (fxp, fyp) = face_center
    return (fxp, fyp)

# Function to get the stable face rotation ratio, LFRR and RFRR 
def face_rotation_ratio(nose_points, left_cheek_points, right_cheek_points):
    nose_center = np.mean(nose_points, axis=0).astype(int)
    left_cheek_center = np.mean(left_cheek_points, axis=0).astype(int)
    right_cheek_center = np.mean(right_cheek_points, axis=0).astype(int)
    
    # Calculate the distance between the cheeks to normalize the nose to cheek distance
    face_cheeks_distance = np.linalg.norm(left_cheek_center - right_cheek_center)
    
    # LFRR and RFRR
    left_face_rotation_ratio = round( np.linalg.norm(left_cheek_center - nose_center) / face_cheeks_distance, 2)
    right_face_rotation_ratio = round(np.linalg.norm(right_cheek_center - nose_center) / face_cheeks_distance , 2)
    
    return (left_face_rotation_ratio, right_face_rotation_ratio) 

# Fucntion to get the stable upper lip to nose ratio
def mouth_nose_aspect_ratio(nose_points, upper_lip_points):
    nose_center = np.mean(nose_points, axis=0).astype(int)
    upper_lip_center = np.mean(upper_lip_points, axis=0).astype(int)
    
    # mouth horizontal distance to normalize the ratio
    mouth_distance = np.linalg.norm(upper_lip_points[0] - upper_lip_points[6]) # P49 - P55
    
    mnar = np.linalg.norm(nose_center - upper_lip_center) / mouth_distance
    return round(mnar, 2)
    
# Function to get the stable lower lip mouth to chin ratio
def mouth_chin_aspect_ratio(lower_lip_points, chin_points):
    chin_center = np.mean(chin_points, axis=0).astype(int)
    lower_lip_center = np.mean(lower_lip_points, axis=0).astype(int)
    
    # lower lip mouth distance to normalize the ratio
    mouth_distance = np.linalg.norm(lower_lip_points[0] - lower_lip_points[4])
    
    mcar = np.linalg.norm(chin_center - lower_lip_center) / mouth_distance
    
    return round(mcar, 2)
    

def main():
    # Load face detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

    # Start video capture (0 - Camera / 1 - Virtual Camera)
    cap = cv2.VideoCapture("C:/Users/Amit/Desktop/HIT/image processing/research/used_videos/AmitS.mkv")

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
            
            # -- EAR -- eye aspect ratio (left ,right)
            # Get eye landmarks (37-42 for left eye, 43-48 for right eye)
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
            
            # Display EAR value
            cv2.putText(frame, f"left EAR: {left_ear}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"right EAR: {right_ear}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # -- IMAR -- inner mouth aspect ratio
            # Inner mouth landmark point (61-68)
            inner_mouth_points = landmarks_points[60:68]

            imar = inner_mouth_aspect_ratio(inner_mouth_points)

            # Draw visualization lines for verticals
            cv2.line(frame, tuple(inner_mouth_points[2]), tuple(inner_mouth_points[6]), (0, 255, 255), 2) # 63 - 67
            cv2.line(frame, tuple(inner_mouth_points[3]), tuple(inner_mouth_points[5]), (0, 255, 255), 2) # 64 - 66
            cv2.line(frame, tuple(inner_mouth_points[1]), tuple(inner_mouth_points[7]), (0, 255, 255), 2) # 62 - 68

            # Draw horizontal line
            cv2.line(frame, tuple(inner_mouth_points[0]), tuple(inner_mouth_points[4]), (255, 0, 255), 1) # 61 - 65

            # Show IMAR on screen
            cv2.putText(frame, f"IMAR: {imar}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # -- EEAR -- eyebrow eye aspect ratio (left, right)
            left_eyebrow = landmarks_points[17:22] # P18 to P22
            right_eyebrow = landmarks_points[22:27] # P23 to P27
            
            left_eyebrow_center = np.mean(left_eyebrow, axis=0).astype(int)
            right_eyebrow_center = np.mean(right_eyebrow, axis=0).astype(int)
            
            # Draw circle on eyebrow center
            cv2.circle(frame, left_eyebrow_center, 2, (255,0,255), 2)
            cv2.circle(frame, right_eyebrow_center, 2, (255,0,255), 2)
            
            # Draw line between eyebrow center to eye center
            cv2.line(frame, left_eyebrow_center, left_eye_center, (255, 0, 0), 2)
            cv2.line(frame, right_eyebrow_center, right_eye_center, (255, 0, 0), 2)

            left_eear = eyebrow_eye_aspect_ratio(left_eyebrow, left_eye)
            right_eear = eyebrow_eye_aspect_ratio(right_eyebrow, right_eye)
            
            # Show EEAR
            cv2.putText(frame, f"left_EEAR: {left_eear}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(frame, f"right_EEAR: {right_eear}", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            
            # -- NP -- nose point (nose x and y coordinates)
            # Calculate the center point of the nose from the mean of nose points 
            nose_points = landmarks_points[27:36] # P28 to P36 
            nose_center = nose_point(nose_points) 
            (nxp, nyp) = nose_center

            cv2.circle(frame, nose_center, 2, (255,0,0), 2)
            cv2.putText(frame, f"NXP: {nxp}", (10,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,0), 2)
            cv2.putText(frame, f"NYP: {nyp}", (10,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,0), 2)
            
            # -- OMAR -- outer mouth aspect ratio
            outer_mouth_points = landmarks_points[48:60] # P49 - P60
            
            omar = outer_mouth_aspect_ratio(outer_mouth_points)
            
            # Show OMAR on screen
            cv2.putText(frame, f"OMAR: {omar}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
            # -- FP -- face point (face x and y coordinates)
            # this feature calculate the mean point of the whole face
            face_points = landmarks_points
            (fxp, fyp) = face_point(face_points)    
            
            # Show FP on screen
            cv2.circle(frame, (fxp, fyp), 2, (0,0,255), 2)
            cv2.putText(frame, f"FXP: {fxp}", (10,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,0), 2)
            cv2.putText(frame, f"FYP: {fyp}", (10,210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,0), 2)
            
            # -- FRR -- face rotation ratio (left and right)
            # For nose center use the already decleared nose_points P28 to P36
            left_cheek_points = landmarks_points[0:5] # P1 - P5
            right_cheek_points = landmarks_points[12 : 17] # P13 - P17
            
            left_cheek_center = np.mean(left_cheek_points, axis=0).astype(int)
            right_cheek_center = np.mean(right_cheek_points, axis=0).astype(int)
            
            (lfrr, rfrr) = face_rotation_ratio(nose_points, left_cheek_points, right_cheek_points)
            
            # show frr on screen
            cv2.line(frame, left_cheek_center, nose_center, (128, 128, 0), 1)
            cv2.line(frame, right_cheek_center, nose_center, (128, 128, 0), 1)
            cv2.putText(frame, f"LFRR: {lfrr}", (10,230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,0), 2)
            cv2.putText(frame, f"RFRR: {rfrr}", (10,250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,0), 2)
            
            # -- MNAR -- mouth nose aspect ratio
            # nose cetner already decleard 
            upper_lip_points = landmarks_points[48:55] # P49 - P55
            upper_lip_center = np.mean(upper_lip_points, axis=0).astype(int)
            
            mnar = mouth_nose_aspect_ratio(nose_points, upper_lip_points)
            
            # show mnar on screen
            cv2.line(frame, nose_center, upper_lip_center, (128,128,128), 2)
            cv2.putText(frame, f"MNAR: {mnar}", (10,270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)
            
            # -- MCAR -- mouth chin aspect ratio
            lower_lip_points = landmarks_points[55:60] # P56 - P60
            chin_points = landmarks_points[7:10] # P8 - P10
            
            lower_lip_center = np.mean(lower_lip_points, axis=0).astype(int)
            chin_center = np.mean(chin_points, axis=0).astype(int)
            
            mcar = mouth_chin_aspect_ratio(lower_lip_points, chin_points)
            
            # show mcar on screen
            cv2.line(frame, lower_lip_center, chin_center, (0,128,128), 2)
            cv2.putText(frame, f"MCAR: {mcar}", (10,290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,128), 2)
                     
        
        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()