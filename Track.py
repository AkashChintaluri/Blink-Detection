import cv2
import dlib
import pyautogui
from imutils import face_utils
from scipy.spatial import distance as dist
import time

pyautogui.FAILSAFE = False

cam = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

screen_w, screen_h = pyautogui.size()
pyautogui.moveTo(screen_w // 2, screen_h // 2)

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

blink_thresh = 0.2
blink_frame_thresh = 5
left_blink_counter = 0
right_blink_counter = 0

sensitivity = 2.0


def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[L_start:L_end]
        right_eye = shape[R_start:R_end]

        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)

        if left_EAR < blink_thresh:
            left_blink_counter += 1
        else:
            if left_blink_counter >= blink_frame_thresh:
                pyautogui.click()
                time.sleep(1)
            left_blink_counter = 0

        if right_EAR < blink_thresh:
            right_blink_counter += 1
        else:
            if right_blink_counter >= blink_frame_thresh:
                pyautogui.click()
                time.sleep(1)
            right_blink_counter = 0

        eye_center_x = int((left_eye[0][0] + left_eye[3][0]) / 2)
        eye_center_y = int((left_eye[0][1] + left_eye[3][1]) / 2)

        screen_x = int(sensitivity * screen_w / frame.shape[1] * eye_center_x)
        screen_y = int(sensitivity * screen_h / frame.shape[0] * eye_center_y)

        screen_x = min(max(screen_x, 0), screen_w)
        screen_y = min(max(screen_y, 0), screen_h)

        pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow('Eye Controlled Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
