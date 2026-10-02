import cv2
import mediapipe as mp
from scipy.spatial import distance
from pygame import mixer
from tkinter import *
from PIL import Image, ImageTk

# ============================================
# SOUND
# ============================================
mixer.init()
mixer.music.load("alert.mp3")

# ============================================
# MEDIAPIPE
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
# THRESHOLDS
# ============================================
EAR_THRESHOLD = 0.22
EAR_FRAME_THRESHOLD = 35

YAWN_THRESHOLD = 0.60
YAWN_FRAME_THRESHOLD = 20

eye_counter = 0
yawn_counter = 0

alarm_on = False
running = False

# ============================================
# CAMERA
# ============================================
cap = None

# ============================================
# FUNCTIONS
# ============================================
def calculate_ear(eye_points):

    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])

    horizontal = distance.euclidean(eye_points[0], eye_points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


def calculate_mar(upper, lower, left, right):

    vertical = distance.euclidean(upper, lower)
    horizontal = distance.euclidean(left, right)

    mar = vertical / horizontal

    return mar


# ============================================
# START DETECTION
# ============================================
def start_detection():

    global running
    global cap

    if running:
        return

    cap = cv2.VideoCapture(camera_index.get())


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    running = True

    update_frame()


# ============================================
# STOP DETECTION
# ============================================
def stop_detection():

    global running
    global cap
    global alarm_on

    running = False

    if cap:
        cap.release()

    mixer.music.stop()

    alarm_on = False

    status_value.config(
        text="STOPPED",
        fg="#808080"
    )

    camera_label.config(image="")


# ============================================
# EXIT APP
# ============================================
def exit_app():

    stop_detection()

    root.destroy()


# ============================================
# UPDATE FRAME
# ============================================
def update_frame():

    global eye_counter
    global yawn_counter
    global alarm_on

    if not running:
        return

    success, frame = cap.read()

    if not success:
        root.after(15, update_frame)
        return

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    status = "ALERT"
    color = "#00C853"

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            left_eye = []
            right_eye = []

            # LEFT EYE
            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                left_eye.append((x, y))

            # RIGHT EYE
            for idx in RIGHT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                right_eye.append((x, y))

            # MOUTH
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

            # ============================================
            # EAR
            # ============================================
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0

            # ============================================
            # MAR
            # ============================================
            mar = calculate_mar(
                upper_lip,
                lower_lip,
                left_mouth,
                right_mouth
            )

            # ============================================
            # DROWSINESS
            # ============================================
            if avg_ear < EAR_THRESHOLD:

                eye_counter += 1

                if eye_counter >= EAR_FRAME_THRESHOLD:

                    status = "DROWSY"
                    color = "#FF1744"

                    if not alarm_on:

                        mixer.music.play(-1)
                        alarm_on = True

            else:
                eye_counter = 0

            # ============================================
            # YAWNING
            # ============================================
            if mar > YAWN_THRESHOLD:

                yawn_counter += 1

                if yawn_counter >= YAWN_FRAME_THRESHOLD:

                    status = "YAWNING"
                    color = "#FF9100"

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
    # STATUS UPDATE
    # ============================================
    status_value.config(
        text=status,
        fg=color
    )

    # Rotate only for DISPLAY
    if camera_index.get() == 1:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ============================================
    # CONVERT IMAGE
    # ============================================
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame)

    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.imgtk = imgtk

    camera_label.configure(image=imgtk)

    # ============================================
    # REFRESH FRAME
    # ============================================
    root.after(15, update_frame)


# ============================================
# ROOT WINDOW
# ============================================
root = Tk()
camera_index = IntVar(value=0)

root.title("AI DROWSINESS DETECTION SYSTEM")

root.geometry("1200x750")

root.configure(bg="#111111")

root.resizable(False, False)

# ============================================
# HEADER
# ============================================
header_frame = Frame(
    root,
    bg="#1A1A1A",
    height=80
)

header_frame.pack(fill=X)

title_label = Label(
    header_frame,
    text="AI Drowsiness Detection System",
    font=("Segoe UI", 24, "bold"),
    bg="#1A1A1A",
    fg="white"
)

title_label.pack(pady=20)

# ============================================
# MAIN FRAME
# ============================================
main_frame = Frame(
    root,
    bg="#111111"
)

main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# ============================================
# CAMERA FRAME
# ============================================
camera_frame = Frame(
    main_frame,
    bg="#1E1E1E",
    width=800,
    height=650,
    highlightbackground="#333333",
    highlightthickness=2
)

camera_frame.grid(row=0, column=0, padx=15)

camera_frame.pack_propagate(False)

camera_label = Label(
    camera_frame,
    bg="#1E1E1E"
)

camera_label.pack(fill=BOTH, expand=True)

# ============================================
# SIDE PANEL
# ============================================
side_panel = Frame(
    main_frame,
    bg="#1A1A1A",
    width=300,
    height=650
)

side_panel.grid(row=0, column=1, sticky="n")

side_panel.pack_propagate(False)

# ============================================
# CAMERA SELECTION
# ============================================
camera_label_title = Label(
    side_panel,
    text="CAMERA SOURCE",
    font=("Segoe UI", 14, "bold"),
    bg="#1A1A1A",
    fg="white"
)

camera_label_title.pack(pady=(40, 10))

camera_options = OptionMenu(
    side_panel,
    camera_index,
    0,
    1,
    2
)

camera_options.config(
    font=("Segoe UI", 12),
    bg="#2A2A2A",
    fg="white",
    width=15,
    bd=0
)

camera_options["menu"].config(
    bg="#2A2A2A",
    fg="white"
)

camera_options.pack()

# ============================================
# STATUS TITLE
# ============================================
status_title = Label(
    side_panel,
    text="USER STATUS",
    font=("Segoe UI", 18, "bold"),
    bg="#1A1A1A",
    fg="white"
)

status_title.pack(pady=(40, 20))

# ============================================
# STATUS VALUE
# ============================================
status_value = Label(
    side_panel,
    text="READY",
    font=("Segoe UI", 28, "bold"),
    bg="#1A1A1A",
    fg="#00C853"
)

status_value.pack(pady=20)

# ============================================
# BUTTONS
# ============================================
button_frame = Frame(
    side_panel,
    bg="#1A1A1A"
)

button_frame.pack(pady=30)

# START BUTTON
start_button = Button(
    button_frame,
    text="START",
    font=("Segoe UI", 14, "bold"),
    bg="#00C853",
    fg="white",
    width=15,
    bd=0,
    activebackground="#00E676",
    cursor="hand2",
    command=start_detection
)

start_button.pack(pady=15)

# STOP BUTTON
stop_button = Button(
    button_frame,
    text="STOP",
    font=("Segoe UI", 14, "bold"),
    bg="#FF9100",
    fg="white",
    width=15,
    bd=0,
    activebackground="#FFAB40",
    cursor="hand2",
    command=stop_detection
)

stop_button.pack(pady=15)

# EXIT BUTTON
exit_button = Button(
    button_frame,
    text="EXIT",
    font=("Segoe UI", 14, "bold"),
    bg="#FF1744",
    fg="white",
    width=15,
    bd=0,
    activebackground="#FF5252",
    cursor="hand2",
    command=exit_app
)

exit_button.pack(pady=15)

# ============================================
# FOOTER
# ============================================
footer = Label(
    root,
    text="AI Powered Driver Safety Monitoring",
    font=("Segoe UI", 10),
    bg="#111111",
    fg="#888888"
)

footer.pack(pady=10)

# ============================================
# RUN APP
# ============================================
root.mainloop()