# Posture Perfect

`Posture Perfect` is a Python application crafted to assist you in maintaining optimal head posture while sitting at your computer. Employing computer vision technology with the support of OpenCV and MediaPipe libraries, it tracks your head position in real time. Whenever the application identifies a deviation from the ideal head alignment, it alerts you with a notification and a sound alert, prompting you to adjust your head posture for better ergonomics."
## Features

- Real-time posture tracking using your webcam.
- Audio and visual alerts when poor posture is detected.
- Customizable thresholds for posture deviation.
- Option to toggle between fixed levels and moving averages for posture detection.

## Requirements

To run `Posture Perfect`, you will need:

- Python 3.6 or newer
- OpenCV
- MediaPipe
- NumPy
- SoundDevice

## Setup

1. Ensure you have Python installed on your system. If not, download and install it from [python.org](https://www.python.org/downloads/).

2. Clone this repository or download the `posture_perfect.py` script.

3. Install the required Python libraries. You can install them using pip:

```shell
pip install opencv-python mediapipe numpy sounddevice
```

4. Ensure you have a webcam connected to your computer for the posture tracking to work.

## Usage

To start monitoring your posture, simply run the `posture_perfect.py` script:

```shell
python posture_perfect.py
```

- The application will open a window showing the video feed from your webcam.
- Sit in a posture you consider ideal and press `t` to set the current posture as the target.
- The application will now monitor your posture and alert you if you deviate from this target posture.
- Press `r` to reset the target posture based on your current posture.
- Press `q` to quit the application.

## Additional Notes

- The application uses macOS notifications for alerts. For Windows or Linux, you might need to modify the `send_notification` function accordingly.
- The audio alert is a simple beep sound. You can customize the sound by modifying the `x` variable in the script.
- The application is designed for educational purposes and is not a replacement for professional ergonomics advice.

## Contributions

Contributions are welcome! If you have improvements or bug fixes, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.