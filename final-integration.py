from flask import Flask, render_template, Response
import cv2
import math
import numpy as np
import dlib
import vlc
from imutils import face_utils

app = Flask(__name__)
#face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Initialize VLC player for alerts
alert = vlc.MediaPlayer('focus.mp3')

# Frame thresholds
frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5
close_thresh = 0.3  # Adjust this based on testing

# Variables
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Helper functions for drowsiness detection
def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (2 * euclideanDist(mouth[0], mouth[6])))

def ear(eye):
    return ((euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3])))

def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

def getFaceDirection(shape, size):
    image_points = np.array([shape[33], shape[8], shape[45], shape[36], shape[54], shape[48]], dtype="double")
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
                             (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
    
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return translation_vector[1][0]

# Video streaming generator function
def generate_frames():
    global flag, yawn_countdown, map_flag, map_counter, close_thresh
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects):
            shape = face_utils.shape_to_np(predictor(gray, rects[0]))
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            leftEAR = ear(leftEye)
            rightEAR = ear(rightEye)
            avgEAR = (leftEAR + rightEAR) / 2.0

            if yawn(shape[mStart:mEnd]) > 0.6:
                cv2.putText(frame, "Yawn Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                yawn_countdown = 1

            if avgEAR < close_thresh:
                flag += 1
                if flag >= frame_thresh_1:
                    cv2.putText(frame, "Drowsy Detected", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    alert.play()
            else:
                flag = 0
                alert.stop()

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
