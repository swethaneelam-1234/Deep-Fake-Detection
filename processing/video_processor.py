"""Frame sampling and pretrained-model inference for uploaded videos."""

import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# This is a frame-level deepfake image classifier. The application samples
# several frames and averages their fake scores to make a simple video score.
# It does NOT by itself constitute a dedicated temporal-attention network.
MODEL_ID = os.environ.get(
    "DEEPFAKE_MODEL_ID", "ashish-001/deepfake-detection-using-ViT"
)
FRAMES_TO_SAMPLE = 8
MAX_VIDEO_SECONDS = 90


@lru_cache(maxsize=1)
def load_model():
    """Download/load the processor and model once, then reuse them."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def extract_frames(video_path, wanted_frames=FRAMES_TO_SAMPLE):
    """Read a video and sample up to wanted_frames evenly spaced frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video. It may be corrupted or unsupported.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0:
        capture.release()
        raise ValueError("No readable frames were found in this video.")

    duration = frame_count / fps if fps and fps > 0 else 0
    if duration > MAX_VIDEO_SECONDS:
        capture.release()
        raise ValueError(f"Please upload a video shorter than {MAX_VIDEO_SECONDS} seconds.")

    sample_count = min(wanted_frames, frame_count)
    frame_positions = np.linspace(0, frame_count - 1, num=sample_count, dtype=int)
    frames = []

    for position in frame_positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = capture.read()
        if success:
            # OpenCV reads BGR images; PIL and Transformers expect RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))

    capture.release()

    if not frames:
        raise ValueError("Could not extract frames from this video.")
    return frames


def get_fake_score(logits, model):
    """Convert model logits to a fake-class score using its configured labels."""
    probabilities = torch.softmax(logits, dim=-1)[0]
    labels = getattr(model.config, "id2label", {}) or {}

    fake_index = None
    for index in range(len(probabilities)):
        label = str(labels.get(index, labels.get(str(index), ""))).lower()
        if "fake" in label or "deepfake" in label:
            fake_index = index
            break

    if fake_index is None:
        # This fallback is specific to the selected model card's label order.
        # If you change MODEL_ID, inspect that model's id2label mapping first.
        if len(probabilities) != 2:
            raise ValueError("Could not identify the fake class in this model's labels.")
        fake_index = 0

    return float(probabilities[fake_index].item())


def analyze_video(video_path):
    """Classify sampled frames and aggregate their fake scores into one result."""
    frames = extract_frames(video_path)
    processor, model = load_model()
    frame_fake_scores = []

    with torch.inference_mode():
        for frame in frames:
            inputs = processor(images=frame, return_tensors="pt")
            outputs = model(**inputs)
            frame_fake_scores.append(get_fake_score(outputs.logits, model))

    # A simple average is easy to understand and gives one video-level score.
    # It is score aggregation, not a learned temporal attention module.
    fake_score = float(np.mean(frame_fake_scores))
    prediction = "Likely Fake" if fake_score >= 0.5 else "Likely Real"

    return {
        "prediction": prediction,
        "fake_score": fake_score,
        "fake_percent": round(fake_score * 100, 1),
        "real_percent": round((1 - fake_score) * 100, 1),
        "frames_checked": len(frames),
        "frame_scores": [round(score * 100, 1) for score in frame_fake_scores],
        "model_id": MODEL_ID,
    }
