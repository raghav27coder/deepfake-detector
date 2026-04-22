"""
DeepFake Detector - Video Processing Module
Extracts and preprocesses frames from uploaded videos using OpenCV.
"""

import cv2
import numpy as np

def crop_face(frame):
    """
    Detect and crop the largest face from the frame.
    If no face is found, returns a center crop of the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    
    if len(faces) > 0:
        # Get the largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        # Add 20% padding around the face
        padding = int(w * 0.2)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        return frame[y_start:y_end, x_start:x_end]
    else:
        # Fallback to center crop
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = w // 2 - min_dim // 2
        start_y = h // 2 - min_dim // 2
        return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]


def extract_frames(video_path, num_frames=20, img_size=(128, 128)):
    """
    Extract evenly spaced frames from a video file.

    Args:
        video_path  : Path to the video file.
        num_frames  : Number of frames to extract.
        img_size    : Tuple (width, height) to resize each frame.

    Returns:
        np.ndarray of shape (num_frames, height, width, 3), or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("[ERROR] Video has 0 frames.")
        cap.release()
        return None

    # Pick evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # Crop face before resizing
        frame = crop_face(frame)
        # Convert BGR → RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("[ERROR] No frames could be read.")
        return None

    # Normalize pixel values to [0, 1]
    frames_array = np.array(frames, dtype=np.float32) / 255.0
    print(f"[INFO] Extracted {len(frames_array)} frames from '{video_path}'")
    return frames_array


def extract_frames_for_training(video_path, num_frames=10, img_size=(128, 128)):
    """
    Lighter version used during dataset preparation for training.
    Returns raw uint8 frames (not normalized).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = crop_face(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size)
        frames.append(frame)

    cap.release()
    return frames
