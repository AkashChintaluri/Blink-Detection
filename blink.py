import cv2  # for video rendering
import dlib  # for face and landmark detection
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os

# Open the webcam
cam = cv2.VideoCapture(0)

# defining a function to calculate the EAR
def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / x1
    return EAR

# Variables
blink_thresh = 0.45
succ_frame = 2
count_frame = 0
blink_start_time = None

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Initializing the Models for Landmark and face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)

    for face in faces:
        shape = landmark_predict(img_gray, face)
        shape = face_utils.shape_to_np(shape)
        lefteye = shape[L_start: L_end]
        righteye = shape[R_start:R_end]
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)
        avg = (left_EAR + right_EAR) / 2

        if avg < blink_thresh:
            if blink_start_time is None:
                blink_start_time = time.time()
            count_frame += 1
        else:
            if blink_start_time is not None:
                blink_duration = time.time() - blink_start_time
                if blink_duration > 2:
                    cv2.putText(frame, 'Blink Detected', (30, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                    os.system('notepad.exe')
                blink_start_time = None
            count_frame = 0

    cv2.imshow("Video", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()