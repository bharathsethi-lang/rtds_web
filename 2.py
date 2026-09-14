# =============================================================================
#  DRIVER BEHAVIOUR MONITORING SYSTEM  —  Single-file version
#  Flow: Config → Vision → Simulation → Preprocessing → Event Detection →
#        Decision → Scoring → Alert → Storage → Apriori → Helpers → Main
# =============================================================================

import cv2
import csv
import math
import os
import random
import struct
import threading
import time
import wave
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import mediapipe as mp

# WebSocket server for the browser frontend
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Always resolve project files relative to this Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# =============================================================================
# 1. CONFIG
# =============================================================================

EAR_THRESHOLD               = 0.25
EAR_CONSEC_FRAMES           = 35
HEAD_ANGLE_THRESHOLD        = 25.0

SPEED_MIN                   = 0.0
SPEED_MAX                   = 120.0
SPEED_LIMIT                 = 80.0
TURN_ANGLE_MAX              = 45.0
SHARP_TURN_THRESHOLD        = 30.0
SIM_NOISE                   = 1.0
DROWSY_LOW_SPEED_THRESHOLD  = 20.0

SMOOTHING_WINDOW            = 5

SPEED_LOW_MAX               = 30.0
SPEED_HIGH_MIN              = 70.0
ACC_HARSH_THRESHOLD         = 5.0

INITIAL_SCORE               = 100
SCORE_DEDUCTIONS            = {
    "drowsy":       40,
    "distracted":   30,
    "overspeed":    20,
    "harsh_brake":  15,
    "sharp_turn":   10,
}
SCORE_RECOVERY_PER_FRAME    = 0.5

PRIORITY_ORDER              = ["drowsy", "distracted", "rash", "aggressive", "safe"]

ALERT_COOLDOWN_FRAMES       = 90
ALERT_SOUND_PATH            = os.path.join(BASE_DIR, "assets", "alarm.wav")

CSV_PATH                    = os.path.join(BASE_DIR, "data", "driver_data.csv")
CSV_COLUMNS                 = [
    "timestamp", "speed", "acceleration", "turn_angle",
    "EAR", "head_angle", "speed_label", "acc_label",
    "events", "score", "status",
]

APRIORI_MIN_SUPPORT         = 0.05
APRIORI_MIN_CONFIDENCE      = 0.5
APRIORI_MIN_LIFT            = 1.0

FONT_SCALE                  = 0.55
OVERLAY_COLOR               = (0, 255, 120)
ALERT_COLOR                 = (0, 0, 255)
WARNING_COLOR               = (0, 165, 255)


# =============================================================================
# 2. VISION
# =============================================================================

VisionResult = namedtuple(
    "VisionResult",
    ["ear", "head_angle", "drowsy", "distracted", "annotated_frame", "face_detected"],
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],
    [0.0,  -330.0, -65.0],
    [-225.0, 170.0,-135.0],
    [225.0,  170.0,-135.0],
    [-150.0,-150.0,-125.0],
    [150.0, -150.0,-125.0],
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _px(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def _ear(landmarks, indices, w, h):
    pts = [_px(landmarks[i], w, h) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-6))

def _head_yaw(landmarks, w, h):
    img_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LANDMARK_IDS], dtype=np.float64)
    cam     = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, img_pts, cam, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy   = math.sqrt(R[0,0]**2 + R[1,0]**2)
    yaw  = math.atan2(R[1,0], R[0,0]) if sy >= 1e-6 else math.atan2(-R[1,2], R[1,1])
    return math.degrees(yaw)


class VisionAnalyser:

    def __init__(self):
        self._mp_face   = mp.solutions.face_mesh
        self._face_mesh = self._mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._mp_draw  = mp.solutions.drawing_utils
        self._draw_spec = self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self._ear_counter = 0

    def vprocess(self, frame):
        h, w   = frame.shape[:2]
        result = self._face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = frame.copy()

        if not result.multi_face_landmarks:
            self._ear_counter = 0
            return VisionResult(ear=0.3, head_angle=0.0, drowsy=False,
                                distracted=False, annotated_frame=annotated, face_detected=False)

        landmarks = result.multi_face_landmarks[0].landmark

        self._mp_draw.draw_landmarks(
            annotated, result.multi_face_landmarks[0],
            self._mp_face.FACEMESH_CONTOURS, self._draw_spec, self._draw_spec,
        )

        ear = (_ear(landmarks, LEFT_EYE, w, h) + _ear(landmarks, RIGHT_EYE, w, h)) / 2.0
        self._ear_counter = self._ear_counter + 1 if ear < EAR_THRESHOLD else 0
        drowsy = self._ear_counter >= EAR_CONSEC_FRAMES

        head_angle = _head_yaw(landmarks, w, h)
        distracted = abs(head_angle) > HEAD_ANGLE_THRESHOLD

        return VisionResult(
            ear=round(ear, 4), head_angle=round(head_angle, 2),
            drowsy=drowsy, distracted=distracted,
            annotated_frame=annotated, face_detected=True,
        )

    def release(self):
        self._face_mesh.close()


# =============================================================================
# 3. SIMULATION
# =============================================================================

DROWSY_ACCEL = "drowsy_accel"
DROWSY_DECEL = "drowsy_decel"


class SensorReading:
    def __init__(self, speed, acceleration, turn_angle):
        self.speed        = speed
        self.acceleration = acceleration
        self.turn_angle   = turn_angle


class VehicleSimulator:

    def __init__(self, fps=10):
        self.fps          = fps
        self.speed        = random.uniform(20, 50)
        self.prev_speed   = self.speed
        self.turn_angle   = 0
        self.drowsy_phase = None

    def step(self, drowsy=False):
        if drowsy:
            delta = self._drowsy_speed_change()
        else:
            self.drowsy_phase = None
            delta = self._normal_speed_change()

        self.speed = max(SPEED_MIN, min(SPEED_MAX, self.speed + delta))
        self._update_turn(drowsy)

        acc = (self.speed - self.prev_speed) / 3.6 / (1 / self.fps)
        self.prev_speed = self.speed

        return SensorReading(
            round(self.speed, 2),
            round(acc, 4),
            round(self.turn_angle, 2),
        )

    def _normal_speed_change(self):
        return random.uniform(-5, 5) + random.gauss(0, SIM_NOISE * 0.1)

    def _drowsy_speed_change(self):
        if self.drowsy_phase is None:
            self.drowsy_phase = random.choice([DROWSY_ACCEL, DROWSY_DECEL])
        if self.drowsy_phase == DROWSY_ACCEL:
            delta = random.uniform(0.3, 1.5)
        else:
            delta = random.uniform(-1.5, -0.2)
        return delta + random.gauss(0, SIM_NOISE * 0.1)

    def _update_turn(self, drowsy):
        drift_chance = 0.06 if drowsy else 0.02
        if random.random() < drift_chance:
            target = random.uniform(-TURN_ANGLE_MAX, TURN_ANGLE_MAX)
            self.turn_angle += (target - self.turn_angle) * 0.3
        else:
            self.turn_angle *= 0.9
        self.turn_angle = max(-TURN_ANGLE_MAX, min(TURN_ANGLE_MAX, self.turn_angle))


# =============================================================================
# 4. PREPROCESSING
# =============================================================================

class ProcessedFrame:
    def __init__(self, speed, acceleration, turn_angle, ear, head_angle,
                 drowsy, distracted, speed_label, acc_label):
        self.speed        = speed
        self.acceleration = acceleration
        self.turn_angle   = turn_angle
        self.ear          = ear
        self.head_angle   = head_angle
        self.drowsy       = drowsy
        self.distracted   = distracted
        self.speed_label  = speed_label
        self.acc_label    = acc_label


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def _mean(buf):
    if buf:
        return sum(buf) / len(buf)
    else:
        return 0.0

def _disc_speed(speed):
    if speed < SPEED_LOW_MAX:  return "low"
    if speed < SPEED_HIGH_MIN: return "medium"
    return "high"

def _disc_acc(acc):
    return "harsh" if abs(acc) >= ACC_HARSH_THRESHOLD else "normal"


class Preprocessor:

    def __init__(self):
        w = SMOOTHING_WINDOW
        self._spd_buf = deque(maxlen=w)
        self._acc_buf = deque(maxlen=w)
        self._trn_buf = deque(maxlen=w)
        self._ear_buf = deque(maxlen=w)

    def process(self, sensor, vision):
        self._spd_buf.append(_clamp(sensor.speed,        SPEED_MIN, SPEED_MAX))
        self._acc_buf.append(_clamp(sensor.acceleration, -30.0, 30.0))
        self._trn_buf.append(_clamp(sensor.turn_angle,   -90.0, 90.0))
        self._ear_buf.append(_clamp(vision.ear,           0.0,  1.0))

        speed        = _mean(self._spd_buf)
        acceleration = _mean(self._acc_buf)
        turn_angle   = _mean(self._trn_buf)
        ear          = _mean(self._ear_buf)

        return ProcessedFrame(
            speed=round(speed, 2),
            acceleration=round(acceleration, 4),
            turn_angle=round(turn_angle, 2),
            ear=round(ear, 4),
            head_angle=round(vision.head_angle, 2),
            drowsy=ear < EAR_THRESHOLD and vision.face_detected,
            distracted=abs(vision.head_angle) > HEAD_ANGLE_THRESHOLD and vision.face_detected,
            speed_label=_disc_speed(speed),
            acc_label=_disc_acc(acceleration),
        )


# =============================================================================
# 5. EVENT DETECTION
# =============================================================================

def detect_events(frame):
    events = []

    if frame.drowsy:
        events.append("drowsy")
        if frame.speed <= 20.0:
            events.append("dangerous_stop")

    if frame.distracted:
        events.append("distracted")

    if frame.speed > SPEED_LIMIT:
        events.append("overspeed")

    if frame.acceleration <= -ACC_HARSH_THRESHOLD:
        events.append("harsh_brake")

    if frame.acceleration >= ACC_HARSH_THRESHOLD:
        events.append("harsh_accel")

    if abs(frame.turn_angle) >= SHARP_TURN_THRESHOLD:
        events.append("sharp_turn")

    return events


# =============================================================================
# 6. DECISION
# =============================================================================

def is_drowsy(evts):
    return "drowsy" in evts

def is_distracted(evts):
    return "distracted" in evts

def is_rash(evts):
    return any(e in evts for e in ("overspeed", "sharp_turn"))

def is_aggressive(evts):
    return any(e in evts for e in ("harsh_brake", "harsh_accel"))

def is_safe(evts):
    return True

_RULES = [
    ("DROWSY",      is_drowsy),
    ("DISTRACTED",  is_distracted),
    ("RASH",        is_rash),
    ("AGGRESSIVE",  is_aggressive),
    ("SAFE",        is_safe),
]

def classify_status(events):
    for status, predicate in _RULES:
        if predicate(events):
            return status
    return "SAFE"


# =============================================================================
# 7. SCORING
# =============================================================================

class RiskScorer:

    def __init__(self):
        self._score = float(INITIAL_SCORE)

    def score(self):
        return self._score

    def update(self, events):
        if events:
            self._score -= sum(SCORE_DEDUCTIONS.get(e, 0) for e in events) * 0.04
        else:
            self._score += SCORE_RECOVERY_PER_FRAME
        self._score = max(0.0, min(float(INITIAL_SCORE), self._score))
        return round(self._score, 1)

    def reset(self):
        self._score = float(INITIAL_SCORE)


# =============================================================================
# 8. ALERT
# =============================================================================

_ALERT_STATUSES       = {"DROWSY", "DISTRACTED"}
DROWSY_SUSTAIN_FRAMES = 210


class AlertManager:

    def __init__(self, play_sound=True):
        self._play_sound_enabled = play_sound
        self._cooldown_counter = 0
        self._playsound_ok     = self._check_playsound()
        self._alert_active     = False
        self._drowsy_counter   = 0

    def evaluate(self, status, frame_number):
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        if status == "DROWSY":
            self._drowsy_counter += 1
            if self._drowsy_counter >= DROWSY_SUSTAIN_FRAMES and self._cooldown_counter == 0:
                self._trigger(status, frame_number)
                self._cooldown_counter = ALERT_COOLDOWN_FRAMES
                self._alert_active = True
                return True
            return False

        self._drowsy_counter = 0

        if status in _ALERT_STATUSES and self._cooldown_counter == 0:
            self._trigger(status, frame_number)
            self._cooldown_counter = ALERT_COOLDOWN_FRAMES
            self._alert_active = True
            return True

        if status not in _ALERT_STATUSES:
            self._alert_active = False

        return False

    @property
    def alert_active(self):
        return self._alert_active

    def drowsy_countdown(self):
        return max(0, DROWSY_SUSTAIN_FRAMES - self._drowsy_counter)

    def _trigger(self, status, frame_number):
        if not self._play_sound_enabled:
            return
        if self._playsound_ok:
            threading.Thread(target=_play_sound, args=(ALERT_SOUND_PATH,), daemon=True).start()
        else:
            print("\a", end="", flush=True)

    @staticmethod
    def _check_playsound():
        try:
            import pygame
            return os.path.exists(ALERT_SOUND_PATH)
        except ImportError:
            return False


def _play_sound(path):
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    except Exception:
        pass


# =============================================================================
# 9. STORAGE
# =============================================================================

class DataStorage:

    def __init__(self, path=CSV_PATH):
        self.file   = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
        if self.file.tell() == 0:
            self.writer.writeheader()

    def write(self, frame, events, score, status):
        self.writer.writerow({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "speed":        frame.speed,
            "acceleration": frame.acceleration,
            "turn_angle":   frame.turn_angle,
            "EAR":          frame.ear,
            "head_angle":   frame.head_angle,
            "speed_label":  frame.speed_label,
            "acc_label":    frame.acc_label,
            "events":       "|".join(events) if events else "none",
            "score":        score,
            "status":       status,
        })

    def close(self):
        self.file.close()


# =============================================================================
# 10. APRIORI ANALYSIS
# =============================================================================

def run_apriori():
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    import pandas as pd

    df = pd.read_csv(CSV_PATH)

    transactions = [str(row).split("|") for row in df["events"]]

    te = TransactionEncoder()
    df_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

    freq_items = apriori(df_encoded, min_support=0.1, use_colnames=True)
    if freq_items.empty:
        return {"summary": "No frequent itemsets found!"}

    rules = association_rules(freq_items, metric="confidence", min_threshold=0.4)

    summary  = "\nFrequent Itemsets:\n" + freq_items.to_string(index=False)
    summary += "\n\nRules:\n" + rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False)
    return {"summary": summary}


# =============================================================================
# 11. HELPERS
# =============================================================================

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


_STATUS_COLORS = {
    "SAFE":       (0, 220, 100),
    "AGGRESSIVE": (0, 165, 255),
    "RASH":       (0, 100, 255),
    "DISTRACTED": (0, 0, 255),
    "DROWSY":     (0, 0, 200),
}

def status_color(status):
    return _STATUS_COLORS.get(status, (200, 200, 200))


def draw_overlay(frame, ear, head_angle, speed, score, status, events, alert):
    h, w  = frame.shape[:2]
    color = status_color(status)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (0, 0), (280, 32), (10, 10, 10), -1)
    cv2.putText(frame, "DRIVER MONITOR", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    for label, value, y in [
        ("EAR",        f"{ear:.3f}",              35),
        ("Head Angle", f"{head_angle:+.1f} deg",   60),
        ("Speed",      f"{speed:.1f} km/h",         85),
        ("Score",      f"{score:.0f} / 100",       110),
    ]:
        cv2.putText(frame, f"{label}:", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 8, 118, 260, 10
    bar_fill  = int(bar_w * max(0.0, min(1.0, score / 100.0)))
    bar_color = (0, 220, 100) if score >= 70 else (0, 165, 255) if score >= 40 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), bar_color, -1)

    badge_y = 148
    cv2.rectangle(frame, (4, badge_y - 20), (276, badge_y + 6), color, -1)
    cv2.putText(frame, f"STATUS: {status}", (8, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    ev_y = 170
    cv2.putText(frame, "Events:", (8, ev_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1, cv2.LINE_AA)
    if events:
        for i, evt in enumerate(events):
            cv2.putText(frame, f"  . {evt}", (8, ev_y + 18 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "  (none)", (8, ev_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 100), 1, cv2.LINE_AA)

    if alert:
        cv2.rectangle(frame, (280, h - 40), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, f"! ALERT: {status} !", (300, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.line(frame, (280, 0), (280, h), (60, 60, 60), 1)
    return frame


def generate_alarm_wav(path="assets/alarm.wav", frequency=880.0,
                       duration=1.0, sample_rate=44100, amplitude=0.5):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = int(sample_rate * duration)
    samples = []
    for i in range(n):
        t    = i / sample_rate
        tone = frequency if (i // (sample_rate // 4)) % 2 == 0 else frequency * 1.5
        samples.append(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * tone * t) * 32767)))
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(samples))


# =============================================================================
# 12. MAIN
# =============================================================================

# =============================================================================
# 12. WEB SERVER — Browser Camera → WebSocket → MediaPipe
# =============================================================================

app = FastAPI(title="DriverGuard Backend")

# -------------------------------------------------------------------------
# FRONTEND FILES
# Everything is resolved relative to 2.py, so `python 2.py` works even if
# the terminal's current directory is different.
# -------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/index.html")
async def serve_index_html():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/styles.css")
async def serve_styles():
    return FileResponse(
        os.path.join(BASE_DIR, "styles.css"),
        media_type="text/css"
    )

@app.get("/app.js")
async def serve_app_js():
    return FileResponse(
        os.path.join(BASE_DIR, "app.js"),
        media_type="application/javascript"
    )

app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(BASE_DIR, "assets")),
    name="assets"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    ensure_dirs(os.path.join(BASE_DIR, "data"), os.path.join(BASE_DIR, "assets"))

    vision = VisionAnalyser()
    simulator = VehicleSimulator()
    pre = Preprocessor()
    scorer = RiskScorer()

    # The browser is responsible for playing the alarm.
    alerter = AlertManager(play_sound=False)
    storage = DataStorage()

    frame_number = 0

    try:
        while True:
            # Browser sends a JPEG frame as binary WebSocket data.
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            frame_bytes = message.get("bytes")

            if not frame_bytes:
                continue

            # Decode JPEG bytes into an OpenCV BGR frame.
            np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({
                    "error": "Could not decode camera frame"
                })
                continue

            frame_number += 1

            # The original DriverGuard pipeline is kept unchanged:
            # Vision → Simulation → Preprocessing → Events → Decision → Score → Alert → Storage
            vis = vision.vprocess(frame)
            sensor = simulator.step(drowsy=vis.drowsy)
            data = pre.process(sensor, vis)
            events = detect_events(data)
            status = classify_status(events)
            score = scorer.update(events)

            alert_triggered = alerter.evaluate(status, frame_number)

            storage.write(data, events, score, status)

            # Send only the data the website needs.
            await websocket.send_json({
                "speed": data.speed,
                "acceleration": data.acceleration,
                "turn_angle": data.turn_angle,
                "ear": data.ear,
                "head_angle": data.head_angle,
                "status": status,
                "score": score,
                "events": events,
                "face_detected": vis.face_detected,
                "alert": alerter.alert_active,
                "alert_triggered": alert_triggered,
                "drowsy_countdown": alerter.drowsy_countdown(),
            })

    except WebSocketDisconnect:
        pass
    finally:
        vision.release()
        storage.close()


def main():
    print("==============================================")
    print("        DRIVERGUARD WEB BACKEND")
    print("==============================================")
    print("WebSocket: ws://localhost:8000/ws")
    print("Open your DriverGuard frontend and start the simulation.")
    print("Press CTRL+C to stop the server.")
    print("==============================================")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )



if __name__ == "__main__":
    main()