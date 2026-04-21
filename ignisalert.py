import cv2
import requests
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv
import os

# ============================================================
# CONFIGURATION — Fill these in before running
# ============================================================

load_dotenv()
SENDER_EMAIL    = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL  = os.getenv("RECEIVER_EMAIL")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

MODEL_PATH      = "best.pt"                  # path to your trained weights
LOCATION_LAT    = 28.6139                    # your forest/camera latitude (Delhi default)
LOCATION_LON    = 77.2090                    # your forest/camera longitude

CONFIDENCE_THRESHOLD = 0.35                  # lower = more sensitive detections
ALERT_COOLDOWN       = 60                    # seconds between repeated alerts

# ============================================================
# RISK LEVELS per detected class
# ============================================================

DETECTION_RISK = {
    "fire":           (4, "CRITICAL",  "Active fire detected!"),
    "smoke":          (3, "HIGH",      "Smoke detected — possible fire nearby"),
    "smoke_wisp":     (2, "MEDIUM",    "Faint smoke detected — monitor closely"),
    "dry_vegetation": (1, "LOW",       "Dry vegetation detected — elevated risk"),
}

# ============================================================
# EMAIL ALERT
# ============================================================

def send_email_alert(subject, message, snapshot_path=None):
    try:
        msg = MIMEMultipart()
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
# WEATHER RISK SCORE
# ============================================================

def get_weather_risk():
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={LOCATION_LAT}&lon={LOCATION_LON}&appid={WEATHER_API_KEY}")
        data = requests.get(url, timeout=10).json()

        temp     = round(data["main"]["temp"] - 273.15, 1)
        humidity = data["main"]["humidity"]
        wind     = data["wind"]["speed"]
        desc     = data["weather"][0]["description"].title()

        score = 0
        if temp > 40:        score += 4
        elif temp > 35:      score += 3
        elif temp > 30:      score += 1

        if humidity < 15:    score += 4
        elif humidity < 25:  score += 3
        elif humidity < 40:  score += 1

        if wind > 15:        score += 3
        elif wind > 10:      score += 2
        elif wind > 6:       score += 1

        if score >= 9:       risk_label = "CRITICAL"
        elif score >= 6:     risk_label = "HIGH"
        elif score >= 3:     risk_label = "MODERATE"
        else:                risk_label = "LOW"

        return {
            "score":    score,
            "label":    risk_label,
            "temp":     temp,
            "humidity": humidity,
            "wind":     wind,
            "desc":     desc
        }

    except Exception as e:
        print(f"[WEATHER ERROR] {e}")
        return None

# ============================================================
# COMBINED RISK ASSESSMENT
# ============================================================

def assess_combined_risk(detections, weather):
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

    weather_score = weather["score"] if weather else 0
    combined      = (max_det_score * 2) + (weather_score // 3)

    if combined >= 8:    final = "CRITICAL"
    elif combined >= 5:  final = "HIGH"
    elif combined >= 2:  final = "MODERATE"
    else:                final = "LOW"

    return final, max_det_label, max_det_msg

# ============================================================
# BUILD EMAIL SUBJECT + BODY
# ============================================================

def build_email(final_risk, det_label, det_msg, detections, weather, frame_count):
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

"""

    if weather:
        w_emoji = emoji_map.get(weather['label'], "🌤")
        body += f"""------------------------------------------------------------
  WEATHER CONDITIONS
------------------------------------------------------------
  Temperature   : {weather['temp']}°C
  Humidity      : {weather['humidity']}%
  Wind Speed    : {weather['wind']} m/s
  Condition     : {weather['desc']}
  Weather Risk  : {w_emoji} {weather['label']}

"""

    body += """------------------------------------------------------------
  A snapshot of the detection has been attached to this email.
  Please verify the situation and take appropriate action.

  — IgnisAlert Automated Alert System
============================================================
"""
    return subject, body

# ============================================================
# MAIN DETECTION LOOP
# ============================================================

def run(source=0):
    """
    source = 0             -> webcam (live)
    source = "video.mp4"   -> video file
    source = "img.jpg"     -> single image
    """
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Refresh weather every 10 minutes
        if time.time() - last_weather_time > 600:
            weather           = get_weather_risk()
            last_weather_time = time.time()

        # Run YOLO detection
        results = model.predict(
                    frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    save=True,
                    project=r"C:\Users\Akshat Salgotra\OneDrive\Desktop\ignisAlert\runs",
                    name="detections"
                )
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

        # Assess combined risk
        final_risk, det_label, det_msg = assess_combined_risk(detections, weather)
        risk_color = risk_colors.get(final_risk, (255, 255, 255))

        cv2.putText(frame, f"Risk: {final_risk}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

        if weather:
            cv2.putText(frame,
                        f"Temp:{weather['temp']}C  Hum:{weather['humidity']}%  Wind:{weather['wind']}m/s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("IgnisAlert", frame)

        # Send email if HIGH or CRITICAL and cooldown passed
        # Send email if HIGH or CRITICAL and cooldown passed
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN:

            # Case 1 — Visual detection is HIGH or CRITICAL
            if final_risk in ("CRITICAL", "HIGH") and detections:
                snapshot_path = f"snapshot_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)

                subject, body = build_email(
                    final_risk, det_label, det_msg,
                    detections, weather, frame_count
                )
                send_email_alert(subject, body, snapshot_path)
                last_alert_time = now

            # Case 2 — No detection but weather alone is dangerous
            elif weather and weather["label"] in ("CRITICAL", "HIGH") and not detections:
                snapshot_path = f"snapshot_{frame_count}.jpg"
                cv2.imwrite(snapshot_path, frame)

                subject, body = build_email(
                    weather["label"], None,
                    "High-risk weather conditions — no active fire visible but monitor closely",
                    [], weather, frame_count
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
    run(source=r"C:\Users\Akshat Salgotra\OneDrive\Desktop\ignisAlert\test_images\test_image.jpg")   # change to 0 for webcam, or "video.mp4"