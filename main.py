import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import time

# ============================================
# INITIALIZE SOUND
# ============================================
mixer.init()
mixer.music.load("alert.mp3")

# ============================================
# MEDIAPIPE FACE MESH
# ============================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# WEBCAM SETUP
# ============================================
cap = cv2.VideoCapture(0)

# Lower resolution for smoother performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================
# EYE LANDMARKS
# ============================================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ============================================
# MOUTH LANDMARKS
# ============================================
UPPER_LIP = 13
LOWER_LIP = 14

LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# ============================================
# SETTINGS
# ============================================
EAR_THRESHOLD = 0.22
EAR_FRAME_THRESHOLD = 35

YAWN_THRESHOLD = 0.60
YAWN_FRAME_THRESHOLD = 20

eye_counter = 0
yawn_counter = 0

alarm_on = False

# FPS calculation
prev_time = 0

# ============================================
# EAR FUNCTION
# ============================================
def calculate_ear(eye_points):

    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])

    horizontal = distance.euclidean(eye_points[0], eye_points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear

# ============================================
# MAR FUNCTION
# ============================================
def calculate_mar(upper, lower, left, right):

    vertical = distance.euclidean(upper, lower)
    horizontal = distance.euclidean(left, right)

    mar = vertical / horizontal

    return mar

# ============================================
# MAIN LOOP
# ============================================
while True:

    success, frame = cap.read()

    if not success:
        print("Failed to access camera")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    # ============================================
    # FACE DETECTION
    # ============================================
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            left_eye = []
            right_eye = []

            # ============================================
            # LEFT EYE
            # ============================================
            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                left_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # ============================================
            # RIGHT EYE
            # ============================================
            for idx in RIGHT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                right_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # ============================================
            # MOUTH POINTS
            # ============================================
            upper_lip = (
                int(face_landmarks.landmark[UPPER_LIP].x * w),
                int(face_landmarks.landmark[UPPER_LIP].y * h)
            )

            lower_lip = (
                int(face_landmarks.landmark[LOWER_LIP].x * w),
                int(face_landmarks.landmark[LOWER_LIP].y * h)
            )

            left_mouth = (
                int(face_landmarks.landmark[LEFT_MOUTH].x * w),
                int(face_landmarks.landmark[LEFT_MOUTH].y * h)
            )

            right_mouth = (
                int(face_landmarks.landmark[RIGHT_MOUTH].x * w),
                int(face_landmarks.landmark[RIGHT_MOUTH].y * h)
            )

            # Draw mouth points
            cv2.circle(frame, upper_lip, 3, (255, 0, 0), -1)
            cv2.circle(frame, lower_lip, 3, (255, 0, 0), -1)
            cv2.circle(frame, left_mouth, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_mouth, 3, (255, 0, 0), -1)

            # ============================================
            # CALCULATE EAR
            # ============================================
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0

            # ============================================
            # CALCULATE MAR
            # ============================================
            mar = calculate_mar(
                upper_lip,
                lower_lip,
                left_mouth,
                right_mouth
            )

            # ============================================
            # DROWSINESS DETECTION
            # ============================================
            if avg_ear < EAR_THRESHOLD:

                eye_counter += 1

                if eye_counter >= EAR_FRAME_THRESHOLD:

                    cv2.putText(
                        frame,
                        "DROWSINESS DETECTED!",
                        (70, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    if not alarm_on:
                        mixer.music.play(-1)
                        alarm_on = True

            else:
                eye_counter = 0

            # ============================================
            # YAWN DETECTION
            # ============================================
            if mar > YAWN_THRESHOLD:

                yawn_counter += 1

                if yawn_counter >= YAWN_FRAME_THRESHOLD:

                    cv2.putText(
                        frame,
                        "YAWNING DETECTED!",
                        (90, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        3
                    )

                    if not alarm_on:
                        mixer.music.play(-1)
                        alarm_on = True

            else:
                yawn_counter = 0

            # ============================================
            # STOP ALARM
            # ============================================
            if (
                avg_ear >= EAR_THRESHOLD
                and mar <= YAWN_THRESHOLD
            ):

                if alarm_on:
                    mixer.music.stop()
                    alarm_on = False

            # ============================================
            # DISPLAY EAR
            # ============================================
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ============================================
            # DISPLAY MAR
            # ============================================
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # ============================================
    # FPS DISPLAY
    # ============================================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # ============================================
    # SHOW WINDOW
    # ============================================
    cv2.imshow("AI Drowsiness Detection System", frame)

    # ============================================
    # EXIT
    # ============================================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================
cap.release()
cv2.destroyAllWindows()
mixer.music.stop()