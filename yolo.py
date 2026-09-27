"""
Live YOLO Detection Using Your Phone Camera (via IP Webcam app)
------------------------------------------------------------------
This uses a phone IP-camera app to turn your phone into a plain HTTP/MJPEG
video stream on your local WiFi network. OpenCV can open that URL directly
with cv2.VideoCapture(), so no WebRTC/vdo.ninja plumbing is needed at all.

SETUP (Android — "IP Webcam" app, free, no account needed):
    1. Install "IP Webcam" by Pavel Khlebovich from the Play Store.
    2. Open it, scroll down, tap "Start server".
    3. It shows an address like:  http://192.168.1.42:8080
    4. Set STREAM_URL below to that address + "/video", e.g.:
           http://192.168.1.42:8080/video
    5. Make sure your phone and computer are on the SAME WiFi network.

SETUP (iOS — "DroidCam" or "EpocCam"):
    Both provide a similar local network stream URL after you start
    the server in the app — use that as STREAM_URL below. Check the
    app's own screen for the exact address/port it gives you, since it
    varies by app.

Install dependencies:
    pip install opencv-python ultralytics

Run:
    python phone_yolo.py

Controls:
    q   quit
"""

import sys

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency 'opencv-python'. Install it with:\n    pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing dependency 'ultralytics'. Install it with:\n    pip install ultralytics")


# ---------------------------------------------------------------------------
# EDIT THIS to match the address your phone's IP-camera app shows you
# ---------------------------------------------------------------------------
STREAM_URL = "http://192.168.1.35:8080/video"

MODEL_PATH = "yolo26n.pt"   # nano model — fastest, downloads automatically on first run
CONF_THRESHOLD = 0.5


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        sys.exit(
            f"Could not connect to phone stream at:\n    {STREAM_URL}\n\n"
            "Check that:\n"
            "  - The IP-camera app's server is running on your phone\n"
            "  - Your phone and computer are on the same WiFi network\n"
            "  - STREAM_URL in this script matches exactly what the app shows\n"
        )

    print(f"Connected to phone stream: {STREAM_URL}")
    print("Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to phone stream.")
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        # .plot() returns the frame with boxes + labels already drawn on it
        annotated = results[0].plot()

        cv2.imshow("YOLO - Phone Camera", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()