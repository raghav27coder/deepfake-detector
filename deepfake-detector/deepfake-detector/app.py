"""
DeepFake Detector - Flask Backend
Main application entry point
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from video_processing import extract_frames
from predict import predict_deepfake

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """Upload page."""
    return render_template('upload.html')


@app.route('/result')
def result_page():
    """Result page (data passed via query params or JS)."""
    return render_template('result.html')


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """
    Video Upload & Analysis API
    Accepts a video file, saves it, extracts frames,
    runs deepfake detection, and returns the result.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .mp4, .avi, or .mov'}), 400

    # Save file with a unique name to prevent collisions
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(unique_name))
    file.save(filepath)

    # Extract frames from video
    frames = extract_frames(filepath, num_frames=20)
    if frames is None or len(frames) == 0:
        return jsonify({'error': 'Could not extract frames from video'}), 500

    # Run deepfake prediction
    result = predict_deepfake(frames)

    return jsonify({
        'label': result['label'],           # "REAL" or "FAKE"
        'confidence': result['confidence'], # e.g. 92.5
        'video_url': f'/uploads/{unique_name}'
    })


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded video files for preview."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─── Run Server ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("[INFO] DeepFake Detector running at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
