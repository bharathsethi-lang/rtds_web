const state = {
  running: false,
  paused: false,
  speed: 0,
  score: 100,
  ear: 0.31,
  head: 3.2,
  acc: 0,
  turn: 0,
  demo: false,
  stream: null,
  timer: null,
  socket: null,
  canvas: null,
  canvasContext: null,
  lastFrameTime: 0
};

const $ = (id) => document.getElementById(id);

const pages = document.querySelectorAll(".page");
const navLinks = document.querySelectorAll(".nav-link");

function goToPage(pageId) {
  pages.forEach(p => p.classList.toggle("active", p.id === pageId));
  navLinks.forEach(n => n.classList.toggle("active", n.dataset.page === pageId));
  window.scrollTo({top: 0, behavior: "smooth"});
}

document.querySelectorAll("[data-page]").forEach(el => {
  el.addEventListener("click", () => goToPage(el.dataset.page));
});

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showAlert("CAMERA UNAVAILABLE", "Use localhost or HTTPS to enable browser camera access.");
    return false;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    $("camera").srcObject = state.stream;
    $("camera").style.display = "block";
    $("cameraPlaceholder").style.display = "none";
    document.querySelector(".scanline").style.display = "block";

    // Hidden canvas used to convert camera frames into JPEGs for Python.
    state.canvas = document.createElement("canvas");
    state.canvas.width = 640;
    state.canvas.height = 480;
    state.canvasContext = state.canvas.getContext("2d");

    return true;
  } catch (err) {
    console.error("Camera error:", err);
    showAlert(
      "CAMERA PERMISSION",
      "Camera access was not granted. Please allow camera access and start again."
    );
    return false;
  }
}

function connectWebSocket() {
  return new Promise((resolve, reject) => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socketUrl = `${protocol}//${window.location.host}/ws`;

    state.socket = new WebSocket(socketUrl);
    state.socket.binaryType = "arraybuffer";

    state.socket.onopen = () => {
      console.log("DriverGuard WebSocket connected.");
      $("headerStatus").textContent = "LIVE";
      resolve(true);
    };

    state.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.error) {
          console.error("Backend:", data.error);
          return;
        }

        updateFromBackend(data);
      } catch (err) {
        console.error("Invalid backend response:", err);
      }
    };

    state.socket.onerror = (err) => {
      console.error("WebSocket error:", err);
      reject(err);
    };

    state.socket.onclose = () => {
      console.log("DriverGuard WebSocket disconnected.");
      if (state.running) {
        $("headerStatus").textContent = "OFFLINE";
      }
    };
  });
}

function sendCameraFrame() {
  if (!state.running || state.paused) return;
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  if (!$("camera").videoWidth || !$("camera").videoHeight) return;
  if (!state.canvas || !state.canvasContext) return;

  const now = performance.now();

  // Limit upload rate to roughly 8 FPS so the backend is not flooded.
  if (now - state.lastFrameTime < 125) return;
  state.lastFrameTime = now;

  state.canvasContext.drawImage(
    $("camera"),
    0,
    0,
    state.canvas.width,
    state.canvas.height
  );

  state.canvas.toBlob(
    (blob) => {
      if (!blob) return;
      if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        state.socket.send(blob);
      }
    },
    "image/jpeg",
    0.65
  );
}

function startSimulation() {
  if (state.running) return;

  state.running = true;
  state.paused = false;
  state.demo = false;

  $("startOverlay").classList.add("hidden");
  $("simulationControls").classList.add("show");
  $("headerStatus").textContent = "CONNECTING";
  document.querySelector(".pulse").classList.add("live");
  $("statusValue").textContent = "CONNECTING...";
  $("statusSub").textContent = "Connecting to DriverGuard analysis engine";
  $("statusHud").classList.remove("danger");

  // Browser autoplay policies allow audio after this user click.
  primeAlarm();

  startCamera().then((cameraReady) => {
    if (!cameraReady) {
      stopSimulation();
      return;
    }

    connectWebSocket()
      .then(() => {
        runLoop();
      })
      .catch(() => {
        showAlert(
          "BACKEND UNAVAILABLE",
          "Make sure 2.py is running and the page was opened from http://localhost:8000."
        );
        stopSimulation();
      });
  });
}

function runLoop() {
  if (state.timer) clearInterval(state.timer);

  // Capture frames independently of the UI update rate.
  state.timer = setInterval(sendCameraFrame, 30);
}

function updateFromBackend(data) {
  if (!state.running) return;

  state.speed = Number(data.speed ?? 0);
  state.score = Number(data.score ?? 100);
  state.ear = Number(data.ear ?? 0.3);
  state.head = Number(data.head_angle ?? 0);
  state.acc = Number(data.acceleration ?? 0);
  state.turn = Number(data.turn_angle ?? 0);

  const events = Array.isArray(data.events) ? data.events : [];
  const status = data.status || "SAFE";

  const flags = {
    overspeed: events.includes("overspeed"),
    drowsy: events.includes("drowsy"),
    distracted: events.includes("distracted"),
    harsh: events.includes("harsh_brake") || events.includes("harsh_accel"),
    sharp: events.includes("sharp_turn")
  };

  let caption = "EXCELLENT";

  if (status === "DROWSY") caption = "TAKE A BREAK";
  else if (status === "DISTRACTED") caption = "FOCUS ON ROAD";
  else if (status === "AGGRESSIVE") caption = "DRIVE SMOOTHLY";
  else if (status === "RASH") caption = "CAUTION";

  renderState(status, caption, flags);

  if (data.alert_triggered) {
    playAlarm();

    if (status === "DROWSY") {
      showAlert(
        "DROWSINESS DETECTED",
        "Please take a break and regain alertness."
      );
    } else if (status === "DISTRACTED") {
      showAlert(
        "DISTRACTION DETECTED",
        "Please focus on the road ahead."
      );
    }
  }
}

function renderState(status, caption, flags) {
  $("speedValue").textContent = Math.round(state.speed).toString().padStart(2, "0");
  $("scoreValue").textContent = Math.round(state.score);
  $("scoreBar").style.width = `${Math.max(0, Math.min(100, state.score))}%`;
  $("scoreCaption").textContent = caption;
  $("earValue").textContent = state.ear.toFixed(2);
  $("headValue").textContent = `${state.head >= 0 ? "+" : ""}${state.head.toFixed(1)}°`;
  $("accValue").textContent = `${state.acc >= 0 ? "+" : ""}${state.acc.toFixed(1)}`;
  $("turnValue").textContent = `${state.turn.toFixed(1)}°`;

  const now = performance.now();
  if (!state.lastRenderTime) state.lastRenderTime = now;
  const elapsed = now - state.lastRenderTime;
  if (elapsed > 500) {
    $("fpsValue").textContent = Math.round(1000 / Math.max(elapsed, 1));
    state.lastRenderTime = now;
  }

  $("statusValue").textContent = status;

  const subs = {
    SAFE: "Driver focused • normal behaviour",
    RASH: "Speed or steering requires attention",
    AGGRESSIVE: "Harsh vehicle input detected",
    DISTRACTED: "Please focus on the road",
    DROWSY: "Drowsiness detected • take a break"
  };

  $("statusSub").textContent = subs[status] || "Monitoring driver behaviour";

  const statusColor = {
    SAFE: "var(--green)",
    RASH: "var(--yellow)",
    AGGRESSIVE: "var(--orange)",
    DISTRACTED: "var(--red)",
    DROWSY: "var(--red)"
  }[status] || "var(--green)";

  $("statusValue").style.color = statusColor;
  document.querySelector(".status-dot").style.background = statusColor;
  document.querySelector(".status-dot").style.boxShadow = `0 0 14px ${statusColor}`;

  setEvent(
    $("eventFace"),
    !flags.drowsy
      ? "✓ FACE / ALERTNESS NORMAL"
      : "⚠ EYES / ALERTNESS LOW",
    flags.drowsy
  );

  setEvent(
    $("eventSpeed"),
    flags.overspeed
      ? `⚠ OVERSPEED ${Math.round(state.speed)} KM/H`
      : "✓ SPEED WITHIN LIMIT",
    flags.overspeed
  );

  if (flags.distracted) {
    setEvent($("eventAttention"), "⚠ DRIVER DISTRACTED", true);
  } else if (flags.sharp) {
    setEvent($("eventAttention"), "⚠ SHARP TURN", true);
  } else {
    setEvent($("eventAttention"), "✓ DRIVER FOCUSED", false);
  }

  // Backend-driven alert banner.
  if (status === "DROWSY") {
    showAlert(
      "DROWSINESS DETECTED",
      "Please take a break and regain alertness."
    );
  } else if (status === "DISTRACTED") {
    showAlert(
      "DISTRACTION DETECTED",
      "Please focus on the road ahead."
    );
  } else if (flags.overspeed) {
    showAlert(
      "OVERSPEED WARNING",
      "Vehicle speed has exceeded the 80 KM/H limit."
    );
  }
}

function setEvent(el, text, danger) {
  el.textContent = text;
  el.className = `event-row ${danger ? "danger" : "good"}`;
}

function showAlert(title, text) {
  $("alertTitle").textContent = title;
  $("alertText").textContent = text;
  $("alertBanner").classList.add("show");
}

function primeAlarm() {
  const audio = $("alarmAudio");

  if (audio) {
    audio.volume = 0.8;

    // Start and immediately pause to unlock audio after the user's click.
    audio.play()
      .then(() => {
        audio.pause();
        audio.currentTime = 0;
      })
      .catch(() => {});
  }
}

function playAlarm() {
  const audio = $("alarmAudio");

  if (!audio) {
    console.warn("alarmAudio element not found.");
    return;
  }

  audio.currentTime = 0;
  audio.volume = 0.8;

  audio.play().catch((err) => {
    console.warn("Browser blocked alarm playback:", err);
  });
}

function togglePause() {
  state.paused = !state.paused;
  $("pauseBtn").textContent = state.paused ? "▶ RESUME" : "Ⅱ PAUSE";
  $("headerStatus").textContent = state.paused ? "PAUSED" : "LIVE";
}

function demoEvent() {
  // Keep the original button, but do not fake backend telemetry.
  showAlert(
    "LIVE BACKEND MODE",
    "Demo telemetry has been replaced by real MediaPipe + DriverGuard analysis."
  );
}

function stopSimulation() {
  state.running = false;
  state.paused = false;
  state.demo = false;

  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }

  if (state.socket) {
    state.socket.close();
    state.socket = null;
  }

  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }

  $("camera").srcObject = null;
  $("camera").style.display = "none";
  $("cameraPlaceholder").style.display = "flex";
  document.querySelector(".scanline").style.display = "none";
  $("simulationControls").classList.remove("show");
  $("startOverlay").classList.remove("hidden");
  $("headerStatus").textContent = "STANDBY";
  document.querySelector(".pulse").classList.remove("live");

  $("statusValue").textContent = "SYSTEM READY";
  $("statusValue").style.color = "var(--green)";
  $("statusSub").textContent = "Start the simulation to begin monitoring";

  state.speed = 0;
  state.score = 100;
  state.ear = 0.31;
  state.head = 3.2;
  state.acc = 0;
  state.turn = 0;

  renderState(
    "SAFE",
    "EXCELLENT",
    {
      overspeed: false,
      drowsy: false,
      distracted: false,
      harsh: false,
      sharp: false
    }
  );
}

$("startBtn").addEventListener("click", startSimulation);
$("pauseBtn").addEventListener("click", togglePause);
$("demoBtn").addEventListener("click", demoEvent);
$("stopBtn").addEventListener("click", stopSimulation);
$("dismissAlert").addEventListener("click", () => {
  $("alertBanner").classList.remove("show");
});

renderState(
  "SAFE",
  "EXCELLENT",
  {
    overspeed: false,
    drowsy: false,
    distracted: false,
    harsh: false,
    sharp: false
  }
);
