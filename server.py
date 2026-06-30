import asyncio
import cv2
import os
import shutil
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path

BASE_DIR    = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ultralytics import YOLO
from ignisalert import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, DRY_VEG_THRESHOLD, ALERT_COOLDOWN,
    detect_dry_vegetation, draw_dry_vegetation_overlay,
    assess_combined_risk, get_weather_risk, send_email_alert, build_email,
)
from db import init_db, log_detection, get_detections, get_stats

# ── Shared state (detection thread → API) ────────────────────

_state = {
    "frame_jpg":        None,
    "risk_level":       "UNKNOWN",
    "det_label":        None,
    "det_classes":      [],
    "dry_veg_ratio":    0.0,
    "dry_veg_detected": False,
    "weather":          None,
    "frame_count":      0,
    "last_alert":       None,
    "running":          False,
    "source":           "none",
}
_lock       = threading.Lock()
_stop_event = threading.Event()
_det_thread = None

LOG_COOLDOWN = 30


# ── Detection thread ─────────────────────────────────────────

def _detection_loop(source):
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        with _lock:
            _state["running"] = False
        return

    risk_colors = {
        "CRITICAL": (0, 0, 255),
        "HIGH":     (0, 128, 255),
        "MODERATE": (0, 215, 255),
        "LOW":      (0, 200, 0),
        "UNKNOWN":  (128, 128, 128),
    }

    os.makedirs(BASE_DIR / "detections", exist_ok=True)

    weather           = get_weather_risk()
    last_alert_time   = 0
    last_weather_time = time.time()
    last_log_time     = 0
    frame_count       = 0
    is_image          = isinstance(source, str) and source.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp"))

    with _lock:
        _state["running"] = True
        _state["weather"] = weather

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_image:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        frame_count += 1

        if time.time() - last_weather_time > 600:
            weather           = get_weather_risk()
            last_weather_time = time.time()

        # Dry vegetation
        dry_veg_detected, dry_veg_ratio, dry_mask = detect_dry_vegetation(frame)
        if dry_veg_detected:
            frame = draw_dry_vegetation_overlay(frame, dry_mask, dry_veg_ratio)

        # YOLO detection
        results    = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                conf_score = float(box.conf[0])
                label      = model.names[cls_id]
                detections.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if label == "fire" else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf_score:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Risk assessment
        final_risk, det_label, det_msg = assess_combined_risk(
            detections, weather, dry_veg_detected
        )
        risk_color = risk_colors.get(final_risk, (255, 255, 255))

        # HUD overlay
        cv2.putText(frame, f"Risk: {final_risk}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
        cv2.putText(frame, f"Frame #{frame_count}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        if weather:
            cv2.putText(frame,
                        f"FWI:{weather['fwi']}  {weather['temp']}C  "
                        f"Hum:{weather['humidity']}%  Wind:{weather['wind']}km/h",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if dry_veg_detected:
            cv2.putText(frame, f"Dry Veg: {dry_veg_ratio*100:.1f}%",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Push to shared state (cast numpy types → Python)
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with _lock:
            _state["frame_jpg"]        = jpg.tobytes()
            _state["risk_level"]       = final_risk
            _state["det_label"]        = det_label
            _state["det_classes"]      = list(set(detections))
            _state["dry_veg_ratio"]    = float(dry_veg_ratio)
            _state["dry_veg_detected"] = bool(dry_veg_detected)
            _state["weather"]          = weather
            _state["frame_count"]      = int(frame_count)

        # Alert logic
        now        = time.time()
        alert_sent = False
        snap_path  = None

        if now - last_alert_time > ALERT_COOLDOWN:
            if final_risk in ("CRITICAL", "HIGH") and detections:
                snap_path = str(BASE_DIR / "detections" / f"snapshot_{frame_count}.jpg")
                cv2.imwrite(snap_path, frame)
                subject, body = build_email(
                    final_risk, det_label, det_msg,
                    detections, weather, frame_count, dry_veg_ratio
                )
                send_email_alert(subject, body, snap_path)
                alert_sent = True

            elif dry_veg_detected and not detections and final_risk in ("MODERATE", "LOW"):
                snap_path = str(BASE_DIR / "detections" / f"snapshot_{frame_count}.jpg")
                cv2.imwrite(snap_path, frame)
                subject, body = build_email(
                    "MODERATE", "LOW",
                    f"Dry vegetation covers {dry_veg_ratio*100:.1f}% — pre-fire conditions",
                    [], weather, frame_count, dry_veg_ratio
                )
                send_email_alert(subject, body, snap_path)
                alert_sent = True

            elif (weather and weather["label"] in ("CRITICAL", "HIGH")
                  and not detections and not dry_veg_detected):
                snap_path = str(BASE_DIR / "detections" / f"snapshot_{frame_count}.jpg")
                cv2.imwrite(snap_path, frame)
                subject, body = build_email(
                    weather["label"], None,
                    f"High FWI ({weather['fwi']}) — dangerous fire weather",
                    [], weather, frame_count, dry_veg_ratio
                )
                send_email_alert(subject, body, snap_path)
                alert_sent = True

            if alert_sent:
                last_alert_time = now
                with _lock:
                    _state["last_alert"] = datetime.now().isoformat()

        # Log notable frames to DB (throttled)
        if (detections or dry_veg_detected) and (now - last_log_time > LOG_COOLDOWN or alert_sent):
            log_detection(
                risk_level    = final_risk,
                det_label     = det_label,
                det_classes   = list(set(detections)),
                dry_veg_ratio = dry_veg_ratio,
                weather       = weather,
                snapshot_path = snap_path,
                alert_sent    = alert_sent,
            )
            last_log_time = now

        time.sleep(0.033 if not is_image else 0.1)

    cap.release()
    with _lock:
        _state["running"] = False
    print("[INFO] Detection loop stopped.")


def _restart_detection(source):
    """Stop the current detection thread and start a new one with `source`."""
    global _det_thread

    _stop_event.set()
    if _det_thread and _det_thread.is_alive():
        _det_thread.join(timeout=5)
    _stop_event.clear()

    # Reset frame so browser doesn't show a stale frozen image
    with _lock:
        _state["frame_jpg"]   = None
        _state["running"]     = False
        _state["risk_level"]  = "UNKNOWN"
        _state["det_classes"] = []
        source_label = "webcam" if source == 0 else Path(source).name
        _state["source"] = source_label

    print(f"[INFO] Starting detection: {source_label}")
    _det_thread = threading.Thread(target=_detection_loop, args=(source,), daemon=True)
    _det_thread.start()


# ── FastAPI app ───────────────────────────────────────────────

os.makedirs(BASE_DIR / "static",     exist_ok=True)
os.makedirs(BASE_DIR / "detections", exist_ok=True)
os.makedirs(UPLOADS_DIR,             exist_ok=True)

app = FastAPI(title="IgnisAlert")

app.mount("/detections", StaticFiles(directory=str(BASE_DIR / "detections")), name="detections")
app.mount("/static",     StaticFiles(directory=str(BASE_DIR / "static")),     name="static")


# ── Video stream ──────────────────────────────────────────────

@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            with _lock:
                jpg = _state["frame_jpg"]
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg + b"\r\n")
            await asyncio.sleep(0.033)
    return StreamingResponse(generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ── Status / data endpoints ───────────────────────────────────

@app.get("/api/status")
def api_status():
    with _lock:
        return {
            "risk_level":       _state["risk_level"],
            "det_label":        _state["det_label"],
            "det_classes":      _state["det_classes"],
            "dry_veg_ratio":    _state["dry_veg_ratio"],
            "dry_veg_detected": _state["dry_veg_detected"],
            "weather":          _state["weather"],
            "frame_count":      _state["frame_count"],
            "last_alert":       _state["last_alert"],
            "running":          _state["running"],
            "source":           _state["source"],
        }


@app.get("/api/detections")
def api_detections(limit: int = 50, offset: int = 0):
    return get_detections(limit, offset)


@app.get("/api/stats")
def api_stats():
    return get_stats()


# ── Source control endpoints ──────────────────────────────────

@app.post("/api/source/webcam")
def source_webcam():
    _restart_detection(0)
    return {"status": "ok", "source": "webcam"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = UPLOADS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _restart_detection(str(dest))
    return {"status": "ok", "source": file.filename}


# ── Dashboard ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(BASE_DIR / "static" / "index.html", encoding="utf-8") as f:
        return f.read()


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IgnisAlert Dashboard Server")
    parser.add_argument("--source", default="none",
                        help="Initial source: 0=webcam, path=file, none=wait for upload")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    init_db()

    if args.source != "none":
        source = int(args.source) if args.source.isdigit() else args.source
        _restart_detection(source)

    print(f"\n✅ IgnisAlert dashboard → http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
