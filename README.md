# 🚗 DriverGuard — Real-Time Driver Safety Monitoring System

> AI-powered real-time driver behaviour monitoring using computer vision and vehicle telemetry.

DriverGuard is a real-time driver safety monitoring system designed to detect potentially dangerous driving behaviour using **computer vision, facial analysis, vehicle telemetry, and event-based risk scoring**.

The system monitors the driver's attention and vehicle behaviour simultaneously, providing a live safety score and alerts when risky behaviour is detected.

---

## ✨ Features

- 👁️ **Drowsiness Detection**
  - Uses Eye Aspect Ratio (EAR)
  - Detects prolonged eye closure
  - Generates drowsiness alerts

- 🧠 **Driver Distraction Detection**
  - Estimates head orientation
  - Detects excessive head movement away from the road

- 🚘 **Vehicle Behaviour Monitoring**
  - Vehicle speed
  - Acceleration / braking
  - Steering / turn angle

- ⚠️ **Risk Event Detection**
  - Drowsiness
  - Distraction
  - Overspeeding
  - Harsh braking
  - Harsh acceleration
  - Sharp turns

- 📊 **Real-Time Safety Score**
  - Starts at 100
  - Score decreases when dangerous events occur
  - Safe driving gradually restores the score

- 🔊 **Driver Alerts**
  - Audio alerts for dangerous behaviour
  - Visual warnings on the dashboard

- 🌐 **Live Web Dashboard**
  - Futuristic in-car cockpit interface
  - Real-time telemetry
  - Driver camera feed
  - Safety status and score
  - Event monitoring

- 🔌 **WebSocket Communication**
  - Browser camera frames are sent to the Python backend
  - Backend processes frames using MediaPipe/OpenCV
  - Real-time telemetry is returned to the dashboard

---

## 🧠 How It Works

```text
┌──────────────────────┐
│   Browser Camera     │
│      Webcam          │
└──────────┬───────────┘
           │ JPEG Frames
           ▼
┌──────────────────────┐
│   FastAPI Backend    │
│      WebSocket       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ OpenCV + MediaPipe   │
│   Face Analysis      │
└──────────┬───────────┘
           │
      ┌────┴─────┐
      ▼          ▼
   EAR / Eyes   Head Angle
      │          │
      └────┬─────┘
           ▼
┌──────────────────────┐
│ Behaviour Detection  │
│ & Event Processing   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Risk Scoring       │
│   + Alert System     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Live Web Dashboard │
│ Speed • Score • HUD  │
└──────────────────────┘