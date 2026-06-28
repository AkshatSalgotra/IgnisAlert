import cv2
import requests
import time
import math
import smtplib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH           = "best.pt"
SENDER_EMAIL         = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD      = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL       = os.getenv("RECEIVER_EMAIL")
WEATHER_API_KEY      = os.getenv("WEATHER_API_KEY")
LOCATION_LAT         = 28.6139
LOCATION_LON         = 77.2090
CONFIDENCE_THRESHOLD = 0.375
ALERT_COOLDOWN       = 60

# Dry vegetation detection threshold
# If more than 30% of frame is yellow-brown, flag as dry vegetation
DRY_VEG_THRESHOLD    = 0.30

# ============================================================
# RISK LEVELS per detected class
# ============================================================

DETECTION_RISK = {
    "fire":           (4, "CRITICAL", "Active fire detected!"),
    "smoke":          (3, "HIGH",     "Smoke detected — possible fire nearby"),
    "smoke_wisp":     (2, "MODERATE", "Faint smoke detected — monitor closely"),
    "dry_vegetation": (1, "LOW",      "Dry vegetation detected — elevated fire risk"),
}

# ============================================================
# DRY VEGETATION DETECTION (Color-based, no training needed)
# Uses HSV color space to detect yellow-brown hues
# typical of dry grass and parched vegetation
# ============================================================

def detect_dry_vegetation(frame):
    """
    Detects dry vegetation using HSV color analysis.
    Dry grass/vegetation appears yellow-brown (hue 15-35 in HSV).
    
    Returns:
        is_dry    : bool — True if significant dry vegetation detected
        ratio     : float — proportion of frame with dry vegetation color
        mask      : the detection mask (for visualization)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow-brown HSV range (dry grass, parched vegetation)
    lower = np.array([15, 40, 40])
    upper = np.array([35, 255, 200])

    mask = cv2.inRange(hsv, lower, upper)

    # Remove small noise using morphological operations
    kernel     = np.ones((5, 5), np.uint8)
    mask       = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask       = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    dry_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    ratio      = dry_pixels / total_pixels

    is_dry = ratio > DRY_VEG_THRESHOLD

    return is_dry, round(ratio, 3), mask


def draw_dry_vegetation_overlay(frame, mask, ratio):
    """
    Draws a subtle green overlay on detected dry vegetation areas
    and shows the coverage percentage on frame.
    """
    # Create colored overlay for dry vegetation areas
    overlay        = frame.copy()
    overlay[mask > 0] = [0, 200, 255]  # yellow-ish tint on dry areas
    frame          = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    # Show dry vegetation percentage
    cv2.putText(frame,
                f"Dry Veg: {ratio*100:.1f}%",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 200, 255), 1)

    return frame

# ============================================================
# EMAIL ALERT
# ============================================================

def send_email_alert(subject, message, snapshot_path=None):
    try:
        msg            = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        if snapshot_path:
            with open(snapshot_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-Disposition", "attachment",
                               filename="ignisalert_snapshot.jpg")
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print(f"[EMAIL SENT] {subject}")

    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# ============================================================
# CANADIAN FOREST FIRE WEATHER INDEX (FWI)
# ============================================================

def calculate_fwi(temp, humidity, wind, rain=0.0):
    mo = 147.2 * (101 - humidity) / (59.5 + humidity)

    if rain > 0.5:
        rf = rain - 0.5
        mr = mo + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
        mo = min(mr, 250)

    ed = (0.942 * (humidity ** 0.679)
          + 11 * math.exp((humidity - 100) / 10)
          + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * humidity)))

    ew = (0.618 * (humidity ** 0.753)
          + 10 * math.exp((humidity - 100) / 10)
          + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * humidity)))

    if mo > ed:
        ko = 0.424 * (1 - (humidity / 100) ** 1.7) + 0.0694 * (wind ** 0.5) * (1 - (humidity / 100) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m  = ed + (mo - ed) * (10 ** (-kd))
    else:
        kl = 0.424 * (1 - ((100 - humidity) / 100) ** 1.7) + 0.0694 * (wind ** 0.5) * (1 - ((100 - humidity) / 100) ** 8)
        kw = kl * 0.581 * math.exp(0.0365 * temp)
        m  = ew - (ew - mo) * (10 ** (-kw))

    ffmc = 59.5 * (250 - m) / (147.2 + m)
    ffmc = max(0, min(ffmc, 101))

    fm   = math.exp(-0.1386 * m) * (1 + (m ** 5.31) / (4.93e7))
    isi  = 0.208 * fm * math.exp(0.05039 * wind)

    dmc  = max(0, 0.5 * temp - 0.3 * humidity + 10)
    dc   = max(0, 0.4 * temp - 0.2 * humidity + 20)
    bui  = (0.8 * dmc * dc) / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0

    if bui <= 80:
        bb = 0.1 * isi * (0.626 * (bui ** 0.809) + 2)
    else:
        bb = 0.1 * isi * (1000 / (25 + 108.64 * math.exp(-0.023 * bui)))

    fwi = bb ** 2.72 if bb > 1 else bb
    fwi = round(fwi, 2)

    if fwi >= 30:    label = "CRITICAL"
    elif fwi >= 17:  label = "HIGH"
    elif fwi >= 10:  label = "MODERATE"
    elif fwi >= 5:   label = "LOW"
    else:            label = "LOW"

    return fwi, label

# ============================================================
# FETCH WEATHER + COMPUTE FWI
# ============================================================

def get_weather_risk():
    try:
        url  = (f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={LOCATION_LAT}&lon={LOCATION_LON}&appid={WEATHER_API_KEY}")
        data = requests.get(url, timeout=10).json()

        temp     = round(data["main"]["temp"] - 273.15, 1)
        humidity = data["main"]["humidity"]
        wind_ms  = data["wind"]["speed"]
        wind_kmh = round(wind_ms * 3.6, 1)
        desc     = data["weather"][0]["description"].title()
        rain     = data.get("rain", {}).get("1h", 0.0)

        fwi_score, fwi_label = calculate_fwi(temp, humidity, wind_kmh, rain)

        print(f"[WEATHER] Temp:{temp}°C  Hum:{humidity}%  Wind:{wind_kmh}km/h  Rain:{rain}mm  FWI:{fwi_score} ({fwi_label})")

        return {
            "temp":     temp,
            "humidity": humidity,
            "wind":     wind_kmh,
            "rain":     rain,
            "desc":     desc,
            "fwi":      fwi_score,
            "label":    fwi_label,
        }

    except Exception as e:
        print(f"[WEATHER ERROR] {e}")
        return None

# ============================================================
# COMBINED RISK ASSESSMENT (Detection + Dry Veg + FWI)
# ============================================================

def assess_combined_risk(detections, weather, dry_veg_detected=False):
    """
    Formula:
        combined = (detection_score * 2) + dry_veg_bonus + fwi_contribution

    Detection weighted double — visual fire/smoke is ground truth.
    Dry vegetation adds +1 as a pre-fire environmental indicator.
    FWI maps weather danger onto 0-4 scale.
    """
    max_det_score = 0
    max_det_label = None
    max_det_msg   = None

    for label in detections:
        key = label.lower()
        if key in DETECTION_RISK:
            score, risk, msg = DETECTION_RISK[key]
            if score > max_det_score:
                max_det_score = score
                max_det_label = risk
                max_det_msg   = msg

    # Dry vegetation adds +1 to combined score
    dry_veg_bonus = 1 if dry_veg_detected else 0

    # If only dry vegetation detected (no fire/smoke), set label
    if dry_veg_detected and max_det_score == 0:
        max_det_label = "LOW"
        max_det_msg   = "Dry vegetation detected — elevated fire risk"

    # Map FWI to 0-4 contribution
    if weather:
        fwi = weather["fwi"]
        if fwi >= 30:    fwi_contribution = 4
        elif fwi >= 17:  fwi_contribution = 3
        elif fwi >= 10:  fwi_contribution = 2
        elif fwi >= 5:   fwi_contribution = 1
        else:            fwi_contribution = 0
    else:
        fwi_contribution = 0

    combined = (max_det_score * 2) + dry_veg_bonus + fwi_contribution

    if combined >= 8:    final = "CRITICAL"
    elif combined >= 5:  final = "HIGH"
    elif combined >= 2:  final = "MODERATE"
    else:                final = "LOW"

    return final, max_det_label, max_det_msg

# ============================================================
# BUILD EMAIL
# ============================================================

def build_email(final_risk, det_label, det_msg, detections,
                weather, frame_count, dry_veg_ratio=0.0):
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    det_str = ", ".join(set(detections)) if detections else "None"

    emoji_map = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MODERATE": "🟡",
        "LOW":      "🟢"
    }
    emoji = emoji_map.get(final_risk, "⚠️")

    subject = f"{emoji} IgnisAlert — {final_risk} Fire Risk Detected [{now}]"

    body = f"""
============================================================
  🔥 IgnisAlert — Wildfire Early Warning System
============================================================

  Time          : {now}
  Frame         : #{frame_count}
  Overall Risk  : {emoji} {final_risk}

------------------------------------------------------------
  VISUAL DETECTION
------------------------------------------------------------
  Detected      : {det_str}
  Risk Level    : {det_label if det_label else 'N/A'}
  Status        : {det_msg if det_msg else 'No threat detected'}

------------------------------------------------------------
  PRE-FIRE CONDITIONS
------------------------------------------------------------
  Dry Vegetation: {'⚠️ YES' if dry_veg_ratio > DRY_VEG_THRESHOLD else '✅ NO'} ({dry_veg_ratio*100:.1f}% of frame)
  Threshold     : {DRY_VEG_THRESHOLD*100:.0f}% coverage triggers warning

"""

    if weather:
        w_emoji = emoji_map.get(weather['label'], "🌤")
        body += f"""------------------------------------------------------------
  WEATHER CONDITIONS (Canadian FWI System)
------------------------------------------------------------
  Temperature   : {weather['temp']}°C
  Humidity      : {weather['humidity']}%
  Wind Speed    : {weather['wind']} km/h
  Rainfall      : {weather['rain']} mm
  Condition     : {weather['desc']}
  FWI Score     : {weather['fwi']}  (0=Low, 5=Mod, 10=High, 17=VHigh, 30+=Extreme)
  Weather Risk  : {w_emoji} {weather['label']}

"""

    body += """------------------------------------------------------------
  A snapshot of the detection has been attached to this email.
  Please verify the situation and take appropriate action.

  — IgnisAlert Automated Alert System
  Risk scoring based on Canadian Forest Fire Weather Index (FWI)
============================================================
"""
    return subject, body

# ============================================================
# MAIN DETECTION LOOP
# ============================================================

def run(source=0):
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print("✅ IgnisAlert is running. Press Q to quit.\n")

    weather           = get_weather_risk()
    last_alert_time   = 0
    last_weather_time = time.time()
    frame_count       = 0

    risk_colors = {
        "CRITICAL": (0, 0, 255),
        "HIGH":     (0, 128, 255),
        "MODERATE": (0, 215, 255),
        "LOW":      (0, 200, 0),
    }

    os.makedirs("detections", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Refresh weather every 10 minutes
        if time.time() - last_weather_time > 600:
            weather           = get_weather_risk()
            last_weather_time = time.time()

        # ── Dry Vegetation Detection ──────────────────────────
        dry_veg_detected, dry_veg_ratio, dry_mask = detect_dry_vegetation(frame)

        if dry_veg_detected:
            frame = draw_dry_vegetation_overlay(frame, dry_mask, dry_veg_ratio)
            print(f"[DRY VEG] Coverage: {dry_veg_ratio*100:.1f}% — Pre-fire risk elevated")

        # ── YOLO Fire/Smoke Detection ─────────────────────────
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
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # Save frame if anything detected
        if detections or dry_veg_detected:
            cv2.imwrite(f"detections/frame_{frame_count}.jpg", frame)

        # ── Combined Risk Assessment ──────────────────────────
        final_risk, det_label, det_msg = assess_combined_risk(
            detections, weather, dry_veg_detected
        )
        risk_color = risk_colors.get(final_risk, (255, 255, 255))

        # Overlay on frame
        cv2.putText(frame, f"Risk: {final_risk}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

        if weather:
            cv2.putText(frame,
                        f"FWI:{weather['fwi']}  Temp:{weather['temp']}C  Hum:{weather['humidity']}%  Wind:{weather['wind']}km/h",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("IgnisAlert", frame)

        # ── Alert Logic ───────────────────────────────────────
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN:

            # Case 1 — Fire/smoke detected (HIGH or CRITICAL)
            if final_risk in ("CRITICAL", "HIGH") and detections:
                snapshot_path = f"detections/snapshot_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)
                subject, body = build_email(
                    final_risk, det_label, det_msg,
                    detections, weather, frame_count, dry_veg_ratio
                )
                send_email_alert(subject, body, snapshot_path)
                last_alert_time = now

            # Case 2 — Dry vegetation detected (no fire/smoke yet)
            elif dry_veg_detected and not detections and final_risk in ("MODERATE", "LOW"):
                snapshot_path = f"detections/snapshot_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)
                subject, body = build_email(
                    "MODERATE", "LOW",
                    f"Dry vegetation covers {dry_veg_ratio*100:.1f}% of frame — pre-fire conditions detected",
                    [], weather, frame_count, dry_veg_ratio
                )
                send_email_alert(subject, body, snapshot_path)
                last_alert_time = now

            # Case 3 — Dangerous weather only (no visual detection)
            elif weather and weather["label"] in ("CRITICAL", "HIGH") and not detections and not dry_veg_detected:
                snapshot_path = f"detections/snapshot_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)
                subject, body = build_email(
                    weather["label"], None,
                    f"High FWI score ({weather['fwi']}) — dangerous fire weather, no active fire visible",
                    [], weather, frame_count, dry_veg_ratio
                )
                send_email_alert(subject, body, snapshot_path)
                last_alert_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ IgnisAlert stopped.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run(source=r"test_images\test2.jpg")   # change to 0 for webcam, or "video.mp4"