"""Deepfake Defender - beginner-friendly Flask application.

This starter version samples video frames and runs a pretrained frame-level
image classifier on them. It aggregates frame scores into a video-level score.
It is a prototype, not a forensic guarantee, and does not yet implement a
separate temporal-attention network or audio/lip-sync analysis.
"""

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from processing.video_processor import analyze_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE_PATH = BASE_DIR / "database.sqlite3"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Load the model only once per running app process. A lock prevents two requests
# from trying to initialize the model simultaneously.
_model_lock = threading.Lock()


def get_db_connection():
    """Open a SQLite connection and return rows that can be read by column name."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the analysis-history table if it does not already exist."""
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fake_score REAL NOT NULL,
                frames_checked INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def allowed_file(filename):
    """Return True when the filename has an extension we accept."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_analysis(filename, result):
    """Save a completed analysis to the local SQLite database."""
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO analyses (filename, prediction, fake_score, frames_checked, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                result["prediction"],
                result["fake_score"],
                result["frames_checked"],
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            ),
        )


@app.route("/")
def home():
    """Show the landing page and the most recent analyses."""
    with get_db_connection() as connection:
        recent_analyses = connection.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT 5"
        ).fetchall()
    return render_template("index.html", recent_analyses=recent_analyses)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Validate the uploaded file, run inference, and show the result."""
    uploaded_file = request.files.get("video")

    if uploaded_file is None or uploaded_file.filename == "":
        flash("Choose a video file before clicking Analyze.", "error")
        return redirect(url_for("home"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Upload MP4, MOV, AVI, MKV, or WEBM.", "error")
        return redirect(url_for("home"))

    original_name = secure_filename(uploaded_file.filename)
    extension = Path(original_name).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{extension}"
    video_path = UPLOAD_DIR / stored_name
    uploaded_file.save(video_path)

    # Check actual file size too; MAX_CONTENT_LENGTH also limits the request.
    if not video_path.exists() or video_path.stat().st_size == 0:
        video_path.unlink(missing_ok=True)
        flash("The uploaded file is empty or could not be saved.", "error")
        return redirect(url_for("home"))

    try:
        with _model_lock:
            result = analyze_video(video_path)
        save_analysis(original_name, result)
        return render_template(
            "result.html",
            filename=original_name,
            result=result,
        )
    except Exception as error:
        # Show a friendly message; detailed diagnostics remain in the terminal.
        app.logger.exception("Video analysis failed")
        flash(
            "Analysis failed. Check the terminal for details. On the first run, "
            "the model needs an internet connection to download its files.",
            "error",
        )
        app.logger.error("Analysis error: %s", error)
        return redirect(url_for("home"))
    finally:
        # Uploaded videos are deleted after analysis. Only the summary is saved.
        video_path.unlink(missing_ok=True)


@app.route("/history")
def history():
    """Show all saved analysis summaries."""
    with get_db_connection() as connection:
        analyses = connection.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return render_template("history.html", analyses=analyses)


@app.errorhandler(413)
def file_too_large(_error):
    flash("That file is too large. The maximum upload size is 50 MB.", "error")
    return redirect(url_for("home"))


initialize_database()

if __name__ == "__main__":
    # Debug is useful while learning. Do not use debug=True on a public server.
    app.run(debug=True, port=5000)
