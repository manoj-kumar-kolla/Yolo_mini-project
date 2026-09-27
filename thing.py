"""
Gesture-Controlled Fireball — v2 (particle/glow renderer + throw physics)
--------------------------------------------------------------------------
Open your hand to hold a fireball. Pinch thumb+index to shrink it, spread
them to grow it. Close your hand fast while moving to THROW it (it flies
under gravity, trails a comet tail, and explodes on impact/expiry). Close
your hand slowly and it just snuffs out in place.

Rendering approach (the same core techniques a real VFX pipeline uses,
just running on the CPU instead of a GPU shader):
    - particles accumulated ADDITIVELY into a float32 buffer, so
      overlapping flame licks build up brightness instead of flattening
    - color-over-life gradient: white-hot core -> yellow -> orange -> red
    - turbulent motion (sinusoidal jitter + buoyancy), not straight lines
    - a bloom pass (bright areas blurred and added back) for glow
    - screen-blended onto the camera frame instead of a hard overwrite

Honest caveat: this is CPU/OpenCV, not a GPU shader pipeline like
Unity/Unreal would use, so there's a ceiling on how far this can go —
but it's built on the same ideas, just real-time on the CPU.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python fireball_gesture.py

Controls:
    q   quit

If it chugs on your machine, lower PARTICLE_CAP and EMIT_RATE below.
"""

import os
import sys
import math
import time
import random

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Missing dependency. Install with:\n    pip install opencv-python numpy")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    sys.exit("Missing dependency 'mediapipe'. Install it with:\n    pip install mediapipe")


# Set this to a phone stream URL (e.g. "http://192.168.1.42:8080/video")
# to use your phone camera instead of the built-in webcam.
VIDEO_SOURCE = 0

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

THUMB_TIP, INDEX_TIP = 4, 8
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

MIN_RADIUS, MAX_RADIUS = 15, 80
THROW_SPEED_THRESHOLD = 550   # px/s hand velocity needed to count as a throw
THROW_MULT = 1.15             # amplify hand velocity slightly for a punchier throw
GRAVITY = 950                 # px/s^2
PARTICLE_CAP = 450
EMIT_RATE = 3                 # flame-lick particles spawned per frame while held


# ---------------------------------------------------------------------------
# Hand tracking setup
# ---------------------------------------------------------------------------
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


def is_fist(landmarks):
    for name, tip_idx in FINGER_TIPS.items():
        pip_idx = FINGER_PIPS[name]
        if landmarks[tip_idx].y < landmarks[pip_idx].y:  # extended
            return False
    return True


# ---------------------------------------------------------------------------
# Additive sprite rendering
# ---------------------------------------------------------------------------
def make_base_sprite(size=64):
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    d = np.sqrt(xx * xx + yy * yy)
    sprite = np.clip(1 - d, 0, 1) ** 2
    return sprite.astype(np.float32)


BASE_SPRITE = make_base_sprite(64)
_sprite_cache = {}


def get_sprite(radius):
    r = max(1, int(round(radius)))
    if r not in _sprite_cache:
        diam = max(2, r * 2)
        _sprite_cache[r] = cv2.resize(BASE_SPRITE, (diam, diam), interpolation=cv2.INTER_LINEAR)
        if len(_sprite_cache) > 300:
            _sprite_cache.clear()
    return _sprite_cache[r]


def splat(buffer, x, y, radius, color, alpha):
    """Additively stamp a soft radial sprite into a float32 BGR buffer."""
    if radius <= 0 or alpha <= 0:
        return
    sprite = get_sprite(radius)
    diam = sprite.shape[0]
    x0, y0 = int(x - diam / 2), int(y - diam / 2)
    x1, y1 = x0 + diam, y0 + diam
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(buffer.shape[1], x1), min(buffer.shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return
    sx0, sy0 = bx0 - x0, by0 - y0
    sx1, sy1 = sx0 + (bx1 - bx0), sy0 + (by1 - by0)
    s = sprite[sy0:sy1, sx0:sx1] * alpha
    region = buffer[by0:by1, bx0:bx1]
    region[..., 0] += s * color[0]
    region[..., 1] += s * color[1]
    region[..., 2] += s * color[2]


# Color-over-life gradient (BGR), t: 1.0 = newborn/hottest -> 0.0 = dead
_FLAME_STOPS = [
    (1.00, (255, 255, 255)),  # white-hot core
    (0.80, (180, 255, 255)),  # pale yellow
    (0.55, (0, 200, 255)),    # orange
    (0.30, (0, 90, 255)),     # red-orange
    (0.10, (0, 20, 120)),     # deep red
    (0.00, (10, 10, 10)),     # smoke
]


def flame_color(t):
    t = max(0.0, min(1.0, t))
    for i in range(len(_FLAME_STOPS) - 1):
        t0, c0 = _FLAME_STOPS[i]
        t1, c1 = _FLAME_STOPS[i + 1]
        if t1 <= t <= t0:
            f = (t - t1) / (t0 - t1 + 1e-6)
            return tuple(c1[k] + (c0[k] - c1[k]) * f for k in range(3))
    return _FLAME_STOPS[-1][1]


# ---------------------------------------------------------------------------
# Particles & projectiles
# ---------------------------------------------------------------------------
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "radius0", "phase")

    def __init__(self, x, y, vx, vy, life, radius0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = self.max_life = life
        self.radius0 = radius0
        self.phase = random.uniform(0, math.tau)

    def update(self, dt, t_now):
        self.vy -= 45 * dt              # buoyancy (image y grows downward)
        self.vx += math.sin(t_now * 4 + self.phase) * 35 * dt  # turbulence
        self.vx *= 0.98
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, buffer):
        t = max(0.0, self.life / self.max_life)
        radius = self.radius0 * (0.3 + 0.7 * t)
        splat(buffer, self.x, self.y, radius, flame_color(t), t ** 0.6)


class Projectile:
    def __init__(self, x, y, vx, vy, radius):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.life = 1.6
        self.alive = True

    def update(self, dt, w, h, particles):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

        for _ in range(2):
            if len(particles) < PARTICLE_CAP:
                particles.append(Particle(
                    self.x + random.uniform(-4, 4), self.y + random.uniform(-4, 4),
                    -self.vx * 0.12 + random.uniform(-40, 40),
                    -self.vy * 0.12 + random.uniform(-40, 40),
                    life=random.uniform(0.25, 0.5), radius0=self.radius * 0.5,
                ))

        if self.life <= 0 or self.x < -60 or self.x > w + 60 or self.y > h + 60:
            self.alive = False

    def draw(self, buffer):
        splat(buffer, self.x, self.y, self.radius * 1.6, (0, 90, 255), 0.30)
        splat(buffer, self.x, self.y, self.radius * 1.1, (0, 160, 255), 0.50)
        splat(buffer, self.x, self.y, self.radius * 0.6, (200, 255, 255), 0.85)


def spawn_explosion(particles, x, y, n=40):
    for _ in range(n):
        if len(particles) >= PARTICLE_CAP:
            break
        ang = random.uniform(0, math.tau)
        speed = random.uniform(80, 380)
        particles.append(Particle(
            x, y, math.cos(ang) * speed, math.sin(ang) * speed - 60,
            life=random.uniform(0.3, 0.7), radius0=random.uniform(6, 14),
        ))


def draw_core(buffer, x, y, r):
    splat(buffer, x, y, r * 1.8, (0, 60, 140), 0.16)
    splat(buffer, x, y, r * 1.3, (0, 110, 200), 0.26)
    splat(buffer, x, y, r * 0.9, (0, 170, 255), 0.42)
    splat(buffer, x, y, r * 0.5, (180, 240, 255), 0.85)


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

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {VIDEO_SOURCE}")

    print("Open hand: hold the fireball. Pinch to resize.")
    print("Fast fist-close while moving: THROW it. Slow fist: snuff it out.")
    print("Press 'q' to quit.")

    particles = []
    projectiles = []

    smooth_x, smooth_y, smooth_r = None, None, 40
    prev_sx, prev_sy = None, None
    holding = False
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = min(0.05, max(1e-3, now - last_time))  # clamp to avoid jumps on stalls
        last_time = now

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        prev_holding = holding
        vel_x, vel_y = 0.0, 0.0
        status = "NO HAND"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            fist_now = is_fist(landmarks)

            if not fist_now:
                thumb, index = landmarks[THUMB_TIP], landmarks[INDEX_TIP]
                cx = (thumb.x + index.x) / 2 * w
                cy = (thumb.y + index.y) / 2 * h
                pinch_dist = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
                target_r = int(np.interp(pinch_dist, [20, 220], [MIN_RADIUS, MAX_RADIUS]))
                target_r = max(MIN_RADIUS, min(MAX_RADIUS, target_r))

                smooth_x = cx if smooth_x is None else 0.6 * smooth_x + 0.4 * cx
                smooth_y = cy if smooth_y is None else 0.6 * smooth_y + 0.4 * cy
                smooth_r = int(0.7 * smooth_r + 0.3 * target_r)

                if prev_sx is not None:
                    vel_x = (smooth_x - prev_sx) / dt
                    vel_y = (smooth_y - prev_sy) / dt
                prev_sx, prev_sy = smooth_x, smooth_y

                holding = True
                status = "HOLDING"
            else:
                holding = False
                if prev_holding and prev_sx is not None:
                    speed = math.hypot(vel_x, vel_y)
                    if speed > THROW_SPEED_THRESHOLD:
                        projectiles.append(Projectile(
                            smooth_x, smooth_y,
                            vel_x * THROW_MULT, vel_y * THROW_MULT,
                            smooth_r,
                        ))
                        status = "THROWN!"
                    else:
                        status = "SNUFFED"
                else:
                    status = "FIST"
                prev_sx, prev_sy = None, None
        else:
            holding = False
            prev_sx, prev_sy = None, None

        buffer = np.zeros_like(frame, dtype=np.float32)

        if holding and smooth_x is not None:
            for _ in range(EMIT_RATE):
                if len(particles) < PARTICLE_CAP:
                    particles.append(Particle(
                        smooth_x + random.uniform(-smooth_r * 0.3, smooth_r * 0.3),
                        smooth_y + random.uniform(-smooth_r * 0.3, smooth_r * 0.3),
                        random.uniform(-20, 20), random.uniform(-90, -30),
                        life=random.uniform(0.3, 0.6), radius0=smooth_r * 0.5,
                    ))
            draw_core(buffer, smooth_x, smooth_y, smooth_r)

        particles = [p for p in particles if p.update(dt, now)]
        for p in particles:
            p.draw(buffer)

        still_alive = []
        for proj in projectiles:
            proj.update(dt, w, h, particles)
            if proj.alive:
                proj.draw(buffer)
                still_alive.append(proj)
            else:
                spawn_explosion(particles, proj.x, proj.y)
        projectiles = still_alive

        # Bloom: blur the bright parts and add them back for a soft glow
        bright = np.clip(buffer - 60, 0, None)
        blurred = cv2.GaussianBlur(bright, (0, 0), sigmaX=12)
        buffer = np.clip(buffer + blurred * 0.5, 0, 255).astype(np.uint8)

        # Screen blend onto the camera frame (keeps highlights hot without
        # just overwriting the image the way a hard alpha blend would)
        frame_f = frame.astype(np.float32)
        buf_f = buffer.astype(np.float32)
        screened = 255 - (255 - frame_f) * (255 - buf_f) / 255.0
        frame = np.clip(screened, 0, 255).astype(np.uint8)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Gesture Fireball", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()