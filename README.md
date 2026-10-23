# Deepfake Defender

A beginner-friendly Flask web application for experimenting with deepfake video detection.

## What this first version does

- Provides a homepage and video upload form.
- Accepts MP4, MOV, AVI, MKV, and WEBM files up to 50 MB and 90 seconds.
- Samples up to 8 frames from a video using OpenCV.
- Runs a pretrained frame-level deepfake image classifier on each sampled frame.
- Averages frame-level fake scores to create a simple video-level score.
- Shows the result and frame scores.
- Stores analysis summaries in SQLite and deletes the uploaded video after analysis.

## Important model limitation

This is a working learning prototype, not a forensic tool. The default model is a frame-level image classifier. Averaging frame predictions is **not** the same as training a temporal-attention network. This version does not analyze audio or verify lip synchronization. Predictions can be wrong, and the score is not a guarantee of authenticity.

The first run downloads model files from Hugging Face and needs an internet connection. The model weights are not included in this repository. The model's training domain and label mapping can affect performance. Review the model card before using the system for research or changing the model ID.

## Requirements

- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for the first model download
- Several GB of free disk space is recommended for Python packages and model files

## Run on macOS

Open this folder in VS Code. Open **Terminal → New Terminal** and run:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If the `pip install` step fails on your Mac, copy the full terminal error and troubleshoot it before proceeding. PyTorch and computer-vision packages can take a while to install.

Open this address in your browser:

```text
http://127.0.0.1:5000
```

Keep the terminal running while you use the website. Press **Control + C** in the terminal to stop the server.

## First analysis

1. Open the site in your browser.
2. Choose a short video (preferably under 30 seconds for the first test).
3. Click **Analyze video**.
4. On the first run, wait while the pretrained model downloads.
5. Review the result and terminal messages if an error occurs.

## Main files

- `app.py`: Flask routes, uploads, and SQLite history.
- `processing/video_processor.py`: frame extraction and model inference.
- `templates/`: HTML pages.
- `static/css/style.css`: page styling.
- `database.sqlite3`: created automatically when the app starts; ignored by Git.
- `uploads/`: temporary video uploads; files are deleted after processing.

## GitHub checklist

Before pushing to GitHub:

- Do not commit uploaded videos, model weights, virtual environments, `.env` files, or `database.sqlite3`.
- Keep `.gitignore` in the repository.
- Do not put private or personal videos in the repository.
- Add your own project screenshots to the README only if they do not contain private data.

## Next improvements

1. Verify predictions on a properly held-out dataset.
2. Add explicit face detection and face cropping.
3. Integrate a dedicated temporal model/attention architecture.
4. Add optional audio-video synchronization analysis.
5. Add authentication and secure per-user analysis history if required.
6. Add automated tests and production configuration.
