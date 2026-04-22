"""
DeepFake Detector - Prediction Module
Multi-signal heuristic detection + optional ML model.
"""

import os
import cv2
import numpy as np

MODEL_PATH = os.path.join('models', 'deepfake_model.h5')
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from tensorflow.keras.models import load_model as keras_load
        print(f"[INFO] Loading model from '{MODEL_PATH}' ...")
        _model = keras_load(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
        return _model
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return None


def _freq_artifact_score(gray):
    """High-freq energy ratio. Deepfakes tend to have unusual FFT patterns."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = 20 * np.log(np.abs(f) + 1)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 6
    Y, X = np.ogrid[:h, :w]
    low_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    lo = np.mean(mag[low_mask])
    hi = np.mean(mag[~low_mask])
    return hi / (lo + 1e-10)


def _face_blending_score(bgr):
    """Color discontinuity at face boundary — blending artifact."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    H, W = bgr.shape[:2]
    diffs = []
    for (fx, fy, fw, fh) in faces:
        m = max(8, int(min(fw, fh) * 0.08))
        face = bgr[fy:fy + fh, fx:fx + fw].astype(np.float32)
        if fy > m:
            above = bgr[fy - m:fy, fx:fx + fw].astype(np.float32)
            diffs.append(np.mean(np.abs(face[:m] - above[-m:])))
        if fy + fh + m < H:
            below = bgr[fy + fh:fy + fh + m, fx:fx + fw].astype(np.float32)
            diffs.append(np.mean(np.abs(face[-m:] - below[:m])))
    return float(np.mean(diffs)) if diffs else None


def _temporal_score(frames):
    """Variance-of-differences between consecutive frames."""
    if len(frames) < 2:
        return 0.0
    diffs = [np.mean(np.abs(frames[i].astype(np.float32) -
                            frames[i - 1].astype(np.float32)))
             for i in range(1, len(frames))]
    return float(np.std(diffs) / (np.mean(diffs) + 1e-10))


def _noise_score(gray):
    """Residual noise variance after Gaussian smoothing."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.var(noise) / (np.mean(np.abs(noise)) + 1e-10))


def predict_deepfake(frames):
    """
    Multi-signal deepfake detection.
    frames: np.ndarray (N, 128, 128, 3) in [0, 1]
    Returns dict with 'label' and 'confidence'.
    """
    # ── Convert frames for OpenCV ────────────────────────────────────────────
    uint8_frames = (frames * 255).astype(np.uint8)
    bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in uint8_frames]
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in uint8_frames]

    weighted_scores = []

    # ── Signal 1: ML model (low weight — trained on synthetic data) ──────────
    model = load_model()
    if model is not None:
        try:
            preds = model.predict(frames, verbose=0)
            ml_score = float(np.mean(preds))
            # Trust the ML model more now that it uses Transfer Learning
            ml_weight = 0.70
            weighted_scores.append((ml_score, ml_weight))
            print(f"[INFO] ML model score: {ml_score:.3f}")
        except Exception as e:
            print(f"[WARN] ML prediction failed: {e}")

    # ── Signal 2: Frequency artifact ────────────────────────────────────────
    freq_scores = [_freq_artifact_score(g) for g in gray_frames]
    avg_freq = float(np.mean(freq_scores))
    # Recalibrated: real videos in this dataset tend to have freq around 0.75 - 0.90
    freq_norm = float(np.clip(abs(avg_freq - 0.80) / 0.30, 0.0, 1.0))
    weighted_scores.append((freq_norm, 0.10))
    print(f"[INFO] Frequency score: {freq_norm:.3f} (raw={avg_freq:.3f})")

    # ── Signal 3: Face blending artifacts ───────────────────────────────────
    blend_raw = [_face_blending_score(b) for b in bgr_frames]
    blend_vals = [v for v in blend_raw if v is not None]
    if blend_vals:
        avg_blend = float(np.mean(blend_vals))
        # Higher discontinuity at boundaries = more fake-like
        blend_norm = float(np.clip(avg_blend / 35.0, 0.0, 1.0))
        weighted_scores.append((blend_norm, 0.10))
        print(f"[INFO] Face blending score: {blend_norm:.3f} (raw={avg_blend:.3f})")
    else:
        # No face detected — rely on other signals
        print("[INFO] No face detected — skipping blending signal.")

    # ── Signal 4: Temporal consistency ──────────────────────────────────────
    temporal = _temporal_score(uint8_frames)
    temporal_norm = float(np.clip(temporal / 1.8, 0.0, 1.0))
    weighted_scores.append((temporal_norm, 0.05))
    print(f"[INFO] Temporal score: {temporal_norm:.3f} (raw={temporal:.3f})")

    # ── Signal 5: Noise pattern ─────────────────────────────────────────────
    noise_vals = [_noise_score(g) for g in gray_frames]
    avg_noise = float(np.mean(noise_vals))
    # Recalibrated: real videos tend to have noise variance ~20-40.
    # We penalize variance < 10 (very smooth, synthetic look)
    noise_norm = float(np.clip(1.0 - (avg_noise / 30.0), 0.0, 1.0))
    weighted_scores.append((noise_norm, 0.05))
    print(f"[INFO] Noise score: {noise_norm:.3f} (raw={avg_noise:.3f})")

    # ── Weighted aggregation ─────────────────────────────────────────────────
    total_w = sum(w for _, w in weighted_scores)
    fake_prob = sum(s * w for s, w in weighted_scores) / (total_w + 1e-10)

    print(f"[INFO] Combined fake probability: {fake_prob:.3f}")

    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = round(
        fake_prob * 100 if label == "FAKE" else (1 - fake_prob) * 100, 2)

    print(f"[INFO] Final: {label} ({confidence:.1f}% confidence)")
    return {'label': label, 'confidence': confidence}
