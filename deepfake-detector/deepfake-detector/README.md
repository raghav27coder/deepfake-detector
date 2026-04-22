# DeepFake Detector 🕵️

A full-stack web application that uses a CNN deep learning model to detect whether a video is real or a deepfake.

Built with **Flask**, **TensorFlow/Keras**, **OpenCV**, and vanilla **HTML/CSS/JS**.

---

## 📁 Project Structure

```
deepfake-detector/
├── app.py                  # Flask server & API routes
├── train_model.py          # Model training script
├── predict.py              # Inference / prediction logic
├── video_processing.py     # OpenCV frame extraction
├── requirements.txt        # Python dependencies
│
├── models/                 # Saved .h5 model goes here
├── uploads/                # Uploaded videos (auto-created)
│
├── dataset/
│   ├── real/               # Place REAL videos here for training
│   └── fake/               # Place FAKE/deepfake videos here
│
├── templates/
│   ├── index.html          # Home page
│   ├── upload.html         # Upload page
│   └── result.html         # Result page
│
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/
```

---

## ⚙️ Installation

### 1. Clone / unzip the project
```bash
cd deepfake-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

### Step 1 — Add training videos

Place video files inside:
- `dataset/real/`  → real face videos (e.g. from FaceForensics++ c0 original sequences)
- `dataset/fake/`  → deepfake videos (e.g. Deepfakes, Face2Face, FaceSwap subsets)

Any `.mp4`, `.avi`, or `.mov` files are accepted. Even 5–10 videos per class will work for a basic demo.

### Step 2 — Run the training script
```bash
python train_model.py
```

The script will:
1. Extract frames from all videos
2. Split into train / validation sets (80/20)
3. Train a CNN model for up to 20 epochs with early stopping
4. Save the best model to `models/deepfake_model.h5`

> **No dataset?** The app runs in **mock mode** — it returns a demo prediction based on frame pixel statistics. This lets you test the full UI without training.

---

## 🚀 Run the Website

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

---

## 🌐 Using the App

1. **Home Page** (`/`) — Learn about deepfakes, click Analyze
2. **Upload Page** (`/upload`) — Drag & drop or browse to select a `.mp4`, `.avi`, or `.mov` file
3. **Loading** — Watch the animated analysis progress
4. **Result Page** (`/result`) — See REAL / FAKE verdict with confidence score and video preview

---

## 📦 Zip the Project

### macOS / Linux
```bash
cd ..
zip -r deepfake-detector.zip deepfake-detector/ --exclude "deepfake-detector/uploads/*" --exclude "deepfake-detector/venv/*" --exclude "deepfake-detector/__pycache__/*"
```

### Windows (PowerShell)
```powershell
Compress-Archive -Path deepfake-detector -DestinationPath deepfake-detector.zip
```

---

## 🔬 Model Architecture

| Layer | Details |
|---|---|
| Conv2D Block 1 | 32 filters, 3×3, ReLU + BN + MaxPool + Dropout(0.25) |
| Conv2D Block 2 | 64 filters, 3×3, ReLU + BN + MaxPool + Dropout(0.25) |
| Conv2D Block 3 | 128 filters, 3×3, ReLU + BN + MaxPool + Dropout(0.25) |
| Conv2D Block 4 | 256 filters, 3×3, ReLU + BN + GlobalAvgPool + Dropout(0.5) |
| Dense | 128 units, ReLU + Dropout(0.5) |
| Output | 1 unit, Sigmoid (0=Real, 1=Fake) |

- **Input size**: 128×128×3  
- **Loss**: Binary Cross-Entropy  
- **Optimizer**: Adam (lr=1e-4)  
- **Callbacks**: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3 |
| ML Framework | TensorFlow 2.16 / Keras |
| Video Processing | OpenCV |
| Frontend | HTML5, CSS3, Vanilla JS |
| Fonts | Syne, Space Mono (Google Fonts) |

---

## 📝 Notes

- Maximum video upload size: **200 MB**
- Uploaded videos are stored temporarily in `uploads/`; you can delete them at any time
- No database is used — results are passed via `sessionStorage`
- The model runs in **mock mode** if `models/deepfake_model.h5` is not present
