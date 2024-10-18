import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils

def euclideanDist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def ear(eye):
    return (euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2.0 * euclideanDist(eye[0], eye[3]))

def getAvg():
    capture = cv2.VideoCapture(0)
    
    # Check if the webcam is accessible
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return None

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # Get the indexes for the left and right eyes
    (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    count = 0.0
    total_ear = 0.0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Frame not captured.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        if len(rects) > 0:
            shape = face_utils.shape_to_np(predictor(gray, rects[0]))
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            
            # Compute EAR for both eyes
            leftEAR = ear(leftEye)
            rightEAR = ear(rightEye)
            total_ear += (leftEAR + rightEAR) / 2.0
            count += 1
            
            # Draw the contours for the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)
        
        cv2.imshow('Training', gray)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break
    
    capture.release()
    cv2.destroyAllWindows()
    
    if count == 0:
        print("No face detected.")
        return None
    return total_ear / count
