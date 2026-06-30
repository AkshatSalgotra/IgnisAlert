import sqlite3
from datetime import datetime

DB_PATH = "ignisalert.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            risk_level    TEXT    NOT NULL,
            det_label     TEXT,
            det_classes   TEXT,
            dry_veg_ratio REAL    DEFAULT 0,
            fwi_score     REAL,
            fwi_label     TEXT,
            temp          REAL,
            humidity      REAL,
            wind          REAL,
            snapshot_path TEXT,
            alert_sent    INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_detection(risk_level, det_label, det_classes, dry_veg_ratio,
                  weather, snapshot_path=None, alert_sent=False):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO detections
            (timestamp, risk_level, det_label, det_classes, dry_veg_ratio,
             fwi_score, fwi_label, temp, humidity, wind, snapshot_path, alert_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        risk_level,
        det_label or "",
        ",".join(det_classes) if det_classes else "",
        dry_veg_ratio,
        weather["fwi"]      if weather else None,
        weather["label"]    if weather else None,
        weather["temp"]     if weather else None,
        weather["humidity"] if weather else None,
        weather["wind"]     if weather else None,
        snapshot_path,
        1 if alert_sent else 0,
    ))
    conn.commit()
    conn.close()


def get_detections(limit=50, offset=0):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    s = {
        "total":       conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
        "critical":    conn.execute("SELECT COUNT(*) FROM detections WHERE risk_level='CRITICAL'").fetchone()[0],
        "high":        conn.execute("SELECT COUNT(*) FROM detections WHERE risk_level='HIGH'").fetchone()[0],
        "alerts_sent": conn.execute("SELECT COUNT(*) FROM detections WHERE alert_sent=1").fetchone()[0],
    }
    conn.close()
    return s
