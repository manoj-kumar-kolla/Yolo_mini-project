"""
Live Sign Language Generator — Phone Camera Source
-----------------------------------------------------
Combines two things we set up separately:
  1. Your phone as the video source, via an IP-camera app (no vdo.ninja/
     WebRTC needed — just a plain HTTP/MJPEG stream OpenCV can open).
  2. Hand-gesture recognition using MediaPipe's modern Tasks API
     (HandLandmarker), which avoids the legacy `mp.solutions.hands`
     AttributeError on newer Python versions.

SETUP (Android — "IP Webcam" app, free, no account needed):
    1. Install "IP Webcam" by Pavel Khlebovich from the Play Store.
    2. Open it, scroll down, tap "Start server".
    3. It shows an address like:  http://192.168.1.42:8080
    4. Set STREAM_URL below to that address + "/video", e.g.:
           http://192.168.1.42:8080/video
    5. Make sure your phone and computer are on the SAME WiFi network.

SETUP (iOS — "DroidCam" or "EpocCam"):
    Same idea — check the app's own screen for its stream address and
    use that as STREAM_URL.

Install dependencies:
    pip install opencv-python mediapipe pyttsx3

Run:
    python phone_sign_language.py

Controls:
    q      quit
    c      clear sentence
    SPACE  add a space
    s      speak the sentence

Recognized gestures: A, B, D, I, L, V, Y (a small rule-based set from
finger geometry — not the full ASL alphabet, see the earlier notes on
why a full alphabet needs a trained model instead of geometric rules).
"""

import os
import sys
import math
import threading

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency 'opencv-python'. Install it with:\n    pip install opencv-python")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    sys.exit("Missing dependency 'mediapipe'. Install it with:\n    pip install mediapipe")

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 150)
    TTS_AVAILABLE = True
except Exception as e:
    print(f"[warning] Text-to-speech unavailable ({e}). 's' key will be disabled.")
    TTS_AVAILABLE = False

_tts_lock = threading.Lock()


def speak(text: str):
    if not TTS_AVAILABLE:
        print("[tts disabled] would have said:", text)
        return

    def _run():
        with _tts_lock:
            _tts_engine.say(text)
            _tts_engine.runAndWait()

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# EDIT THIS to match the address your phone's IP-camera app shows you
# ---------------------------------------------------------------------------
STREAM_URL = "http://192.168.1.35:8080/video"


# ---------------------------------------------------------------------------
# Model download (first run only)
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-tracking model (first run only)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    except Exception as e:
        sys.exit(
            f"Could not download the hand-tracking model automatically ({e}).\n"
            f"Download it manually from:\n    {MODEL_URL}\n"
            f"and place it next to this script as '{MODEL_PATH}'."
        )


# ---------------------------------------------------------------------------
# Hand landmark classification
# ---------------------------------------------------------------------------
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP, THUMB_IP, INDEX_MCP = 4, 3, 5

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),
]


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def finger_states(landmarks):
    state = {}
    for name, tip_idx in FINGER_TIPS.items():
        pip_idx = FINGER_PIPS[name]
        state[name] = landmarks[tip_idx].y < landmarks[pip_idx].y

    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    base = landmarks[INDEX_MCP]
    dist_tip = math.hypot(tip.x - base.x, tip.y - base.y)
    dist_ip = math.hypot(ip.x - base.x, ip.y - base.y)
    state["thumb"] = dist_tip > dist_ip * 1.15

    return state


def classify_letter(state):
    pattern = (state["thumb"], state["index"], state["middle"], state["ring"], state["pinky"])
    gestures = {
        (False, False, False, False, False): "A",  # fist
        (True, True, True, True, True): "B",       # open hand
        (False, True, False, False, False): "D",   # index only
        (False, False, False, False, True): "I",   # pinky only
        (True, True, False, False, False): "L",    # thumb + index (L shape)
        (False, True, True, False, False): "V",    # index + middle (peace)
        (True, False, False, False, True): "Y",    # thumb + pinky (hang loose)
    }
    return gestures.get(pattern, "")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

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

    current_sentence = ""
    recent_predictions = []
    STABILITY_THRESHOLD = 12
    NO_HAND_RESET_FRAMES = 15
    empty_frame_count = 0

    print("Starting Live Sign Language Generator...")
    print("Press 'c' to clear | SPACE for a space | 's' to speak | 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to phone stream.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        detected_letter = ""
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks, w, h)
            state = finger_states(landmarks)
            detected_letter = classify_letter(state)

            if detected_letter:
                cv2.putText(frame, f"Sign: {detected_letter}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if detected_letter:
            empty_frame_count = 0
            recent_predictions.append(detected_letter)
            if len(recent_predictions) > STABILITY_THRESHOLD:
                recent_predictions.pop(0)

            if len(recent_predictions) == STABILITY_THRESHOLD and len(set(recent_predictions)) == 1:
                confirmed = recent_predictions[0]
                if not current_sentence.endswith(confirmed):
                    current_sentence += confirmed
                    recent_predictions.clear()
        else:
            empty_frame_count += 1
            if empty_frame_count >= NO_HAND_RESET_FRAMES:
                recent_predictions.clear()

        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Sentence: {current_sentence}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Sign Language Generator - Phone Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_sentence = ""
            recent_predictions.clear()
        elif key == ord(' '):
            current_sentence += " "
            recent_predictions.clear()
        elif key == ord('s') and current_sentence:
            speak(current_sentence)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()