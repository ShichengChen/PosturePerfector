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
num_frames = 0
threshold = 0.05
start_time = None
use_fixed_levels = False
fixed_eye_level = None
fixed_nose_level = None

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
            else:
                num_frames += 1
                avg_eye_level = (avg_eye_level * (num_frames - 1) + current_eye_level) / num_frames
                avg_nose_level = (avg_nose_level * (num_frames - 1) + current_nose_level) / num_frames
            if use_fixed_levels and current_eye_level <= fixed_eye_level + threshold and current_nose_level <= fixed_nose_level + threshold:
                start_time = time.time()
            if not use_fixed_levels and current_eye_level <= avg_eye_level + threshold and current_nose_level <= avg_nose_level + threshold:
                start_time = time.time()

            comparison_eye_level = fixed_eye_level if use_fixed_levels else avg_eye_level
            comparison_nose_level = fixed_nose_level if use_fixed_levels else avg_nose_level


            # Draw separate baseline and threshold lines for eyes and nose
            def draw_lines(y_value, color, label):
                y_pixel = int(y_value * frame.shape[0])
                cv2.line(frame, (0, y_pixel), (frame.shape[1], y_pixel), color, 2)
                cv2.putText(frame, label, (10, y_pixel - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


            # Eye lines
            draw_lines(comparison_eye_level, (255, 0, 0), "Eye Baseline")
            draw_lines(comparison_eye_level + threshold, (0, 0, 255), "Eye Threshold")

            # Nose lines
            draw_lines(comparison_nose_level, (0, 255, 0), "Nose Baseline")
            draw_lines(comparison_nose_level + threshold, (0, 255, 255), "Nose Threshold")

            # Add captions
            cv2.putText(frame, "Using Fixed Levels" if use_fixed_levels else "Using Moving Average", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Check if threshold is crossed for more than 1 second
            if current_eye_level > comparison_eye_level + threshold or current_nose_level > comparison_nose_level + threshold:
                if time.time() - start_time >= 1:  # 1 seconds
                    sd.play(x, fs)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose', frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('t'):
            fixed_eye_level = avg_eye_level
            fixed_nose_level = avg_nose_level
            use_fixed_levels = True
        elif key & 0xFF == ord('r'):
            use_fixed_levels = False

cap.release()
cv2.destroyAllWindows()
