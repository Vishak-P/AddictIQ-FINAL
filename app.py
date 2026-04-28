"""
app.py
======
Social Media & Mobile Addiction Prediction — Flask Backend
Author  : ML/DevOps Engineer
Purpose : Serve the trained model via a REST API + render the UI.
"""

import os
import json
import pickle
import logging
from functools import lru_cache

import mysql.connector
from mysql.connector import Error
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH   = "model.pkl"
METRICS_PATH = "metrics.json"

# ──────────────────────────────────────────────
# MYSQL CONFIG
# ──────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.environ.get("MYSQL_HOST",     "localhost"),
    "port":     int(os.environ.get("MYSQL_PORT", 3306)),
    "user":     os.environ.get("MYSQL_USER",     "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "addictiq"),
}


def get_db():
    """Open a new MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    """Create database and predictions table if they don't exist."""
    try:
        cfg = {**DB_CONFIG}
        db_name = cfg.pop("database")
        conn = mysql.connector.connect(**cfg)
        cur  = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        conn.database = db_name
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                age              FLOAT NOT NULL,
                daily_usage_hours FLOAT NOT NULL,
                social_media_apps INT  NOT NULL,
                screen_time      FLOAT NOT NULL,
                sleep_hours      FLOAT NOT NULL,
                prediction       VARCHAR(20) NOT NULL,
                confidence       FLOAT NOT NULL,
                addicted_prob    FLOAT NOT NULL,
                not_addicted_prob FLOAT NOT NULL,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialised.")
    except Error as e:
        logger.warning("DB init failed (MySQL may be offline): %s", e)


def save_prediction(features, result):
    """Insert one prediction record into MySQL."""
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO predictions
              (age, daily_usage_hours, social_media_apps, screen_time,
               sleep_hours, prediction, confidence, addicted_prob, not_addicted_prob)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            features[0], features[1], features[2], features[3], features[4],
            result["prediction"], result["confidence"],
            result["addicted_prob"], result["not_addicted_prob"],
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Error as e:
        logger.warning("Could not save prediction to DB: %s", e)


# Feature bounds for input validation
FEATURE_BOUNDS = {
    "Age":               (1,   100),
    "Daily_Usage_Hours": (0.0, 24.0),
    "Social_Media_Apps": (0,   100),
    "Screen_Time":       (0.0, 24.0),
    "Sleep_Hours":       (0.0, 24.0),
}


# ──────────────────────────────────────────────
# MODEL / METRICS LOADING  (cached)
# ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained sklearn pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Run 'python train_model.py' first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


@lru_cache(maxsize=1)
def load_metrics():
    """Load and cache the evaluation metrics JSON."""
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    logger.info("Metrics loaded from %s", METRICS_PATH)
    return metrics


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def validate_inputs(data: dict):
    """
    Validate and parse incoming feature values.
    Returns (features_list, error_message).
    """
    parsed = {}
    for field, (low, high) in FEATURE_BOUNDS.items():
        raw = data.get(field)
        if raw is None:
            return None, f"Missing field: '{field}'"
        try:
            value = float(raw)
        except (ValueError, TypeError):
            return None, f"Field '{field}' must be a number; got '{raw}'"
        if not (low <= value <= high):
            return None, (
                f"Field '{field}' = {value} is out of range [{low}, {high}]"
            )
        parsed[field] = value

    feature_vector = [
        parsed["Age"],
        parsed["Daily_Usage_Hours"],
        parsed["Social_Media_Apps"],
        parsed["Screen_Time"],
        parsed["Sleep_Hours"],
    ]
    return feature_vector, None


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main UI page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  /predict → friendly message
    POST /predict → run prediction
    """
    if request.method == "GET":
        return jsonify({
            "message": "Use POST with JSON body: {Age, Daily_Usage_Hours, Social_Media_Apps, Screen_Time, Sleep_Hours}"
        }), 200
    try:
        # Accept JSON body or form data
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = request.form.to_dict()

        features, error = validate_inputs(data)
        if error:
            return jsonify({"success": False, "error": error}), 400

        model = load_model()
        prediction_code = model.predict([features])[0]          # 0 or 1
        proba           = model.predict_proba([features])[0]     # [p_no, p_yes]

        label            = "Addicted" if prediction_code == 1 else "Not Addicted"
        confidence       = round(float(max(proba)) * 100, 1)
        addicted_prob    = round(float(proba[1]) * 100, 1)

        logger.info(
            "Prediction: %s (confidence=%.1f%%) | inputs=%s",
            label, confidence, features,
        )

        result = {
            "success":           True,
            "prediction":        label,
            "confidence":        confidence,
            "addicted_prob":     addicted_prob,
            "not_addicted_prob": round(float(proba[0]) * 100, 1),
        }

        save_prediction(features, result)

        return jsonify(result)

    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 503

    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    GET /metrics
    Returns model evaluation metrics as JSON.
    """
    try:
        data = load_metrics()
        if not data:
            return jsonify({"error": "Metrics file not found"}), 404
        return jsonify(data)
    except Exception as exc:
        logger.exception("Error loading metrics")
        return jsonify({"error": "Could not load metrics"}), 500


@app.route("/history", methods=["GET"])
def history():
    """
    GET /history
    Returns the last 50 predictions from MySQL.
    """
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, age, daily_usage_hours, social_media_apps,
                   screen_time, sleep_hours, prediction, confidence,
                   addicted_prob, not_addicted_prob,
                   DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') AS created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify({"success": True, "history": rows})
    except Error as e:
        logger.warning("Could not fetch history: %s", e)
        return jsonify({"success": False, "error": "Database unavailable"}), 503


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint for CI/CD and Render."""
    return jsonify({"status": "ok", "service": "addiction-predictor"}), 200


@app.route("/db-check", methods=["GET"])
def db_check():
    """Check MySQL connectivity."""
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT DATABASE(), VERSION()")
        db_name, version = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({
            "status":   "connected",
            "database": db_name,
            "version":  version,
            "host":     DB_CONFIG["host"],
            "port":     DB_CONFIG["port"],
        }), 200
    except Error as e:
        return jsonify({"status": "error", "error": str(e)}), 503


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
