import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time

# Generate a simple beep sound
fs = 44100
seconds = 0.5
t = np.linspace(0, np.pi * 2, int(fs * seconds), endpoint=False)
x = 0.5 * np.sin(440 * 2 * np.pi * t)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

# Initialize variables
avg_eye_level = 0
avg_nose_level = 0
fixed_baseline = 0
num_frames = 0
threshold = 0.05
start_time = None
use_fixed_baseline = False  # New Variable

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_eye_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y
            right_eye_y = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y

            current_eye_level = (left_eye_y + right_eye_y) / 2
            current_nose_level = nose_y

            if start_time is None:
                avg_eye_level = current_eye_level
                avg_nose_level = current_nose_level
                num_frames = 1
                start_time = time.time()
            elif current_eye_level <= avg_eye_level + threshold and current_nose_level <= avg_nose_level + threshold:
                num_frames += 1
                avg_eye_level = (avg_eye_level * (num_frames - 1) + current_eye_level) / num_frames
                avg_nose_level = (avg_nose_level * (num_frames - 1) + current_nose_level) / num_frames
                start_time = time.time()

            if use_fixed_baseline:
                baseline_y = fixed_baseline
            else:
                baseline_y = int(avg_eye_level * frame.shape[0])
                fixed_baseline = baseline_y

            threshold_y = int((avg_eye_level + threshold) * frame.shape[0])

            cv2.line(frame, (0, baseline_y), (frame.shape[1], baseline_y), (255, 0, 0), 2)
            cv2.line(frame, (0, threshold_y), (frame.shape[1], threshold_y), (0, 0, 255), 2)

            cv2.putText(frame, "Baseline", (10, baseline_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, "Threshold", (10, threshold_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if current_eye_level > avg_eye_level + threshold or current_nose_level > avg_nose_level + threshold:
                if time.time() - start_time >= 2:
                    sd.play(x, fs)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose', frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('t'):
            fixed_baseline = int(current_eye_level * frame.shape[0])
            use_fixed_baseline = True
        elif key & 0xFF == ord('r'):
            use_fixed_baseline = False

cap.release()
cv2.destroyAllWindows()
