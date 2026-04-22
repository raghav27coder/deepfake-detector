"""
DeepFake Detector - Model Training Script
==========================================
Trains a CNN-based binary classifier to detect deepfake videos.

Dataset folder structure expected:
    dataset/
        real/   ← .mp4 / .avi / .mov videos of REAL faces
        fake/   ← .mp4 / .avi / .mov videos of FAKE (deepfake) faces

Usage:
    python train_model.py

The trained model is saved to:
    models/deepfake_model.h5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from video_processing import extract_frames_for_training

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE      = (128, 128)
FRAMES_PER_VIDEO = 15
BATCH_SIZE    = 32
EPOCHS        = 10
DATASET_DIR   = 'dataset'
MODEL_SAVE_PATH = os.path.join('models', 'deepfake_model.h5')
REAL_DIR      = os.path.join(DATASET_DIR, 'real')
FAKE_DIR      = os.path.join(DATASET_DIR, 'fake')
VIDEO_EXTS    = {'.mp4', '.avi', '.mov'}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset():
    """
    Load frames from all videos in dataset/real and dataset/fake.
    Returns X (frames array) and y (labels: 0=real, 1=fake).
    """
    X, y = [], []

    for label_int, folder in [(0, REAL_DIR), (1, FAKE_DIR)]:
        label_name = 'REAL' if label_int == 0 else 'FAKE'
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        videos = [f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in VIDEO_EXTS]

        print(f"[INFO] Loading {len(videos)} {label_name} videos from '{folder}' ...")

        for vid_file in videos:
            vid_path = os.path.join(folder, vid_file)
            frames = extract_frames_for_training(vid_path,
                                                 num_frames=FRAMES_PER_VIDEO,
                                                 img_size=IMG_SIZE)
            for frame in frames:
                X.append(frame)
                y.append(label_int)

    if len(X) == 0:
        raise ValueError(
            "No training data found!\n"
            "Place .mp4/.avi/.mov videos inside:\n"
            f"  {REAL_DIR}/  (real videos)\n"
            f"  {FAKE_DIR}/  (fake/deepfake videos)"
        )

    X = np.array(X, dtype=np.float32) / 255.0  # normalize to [0, 1]
    y = np.array(y, dtype=np.float32)
    print(f"[INFO] Total frames loaded: {len(X)}  (real={np.sum(y==0)}, fake={np.sum(y==1)})")
    return X, y


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model(input_shape=(128, 128, 3)):
    """
    Build a CNN deepfake detection model using Transfer Learning.
    Architecture: MobileNetV2 -> GlobalAvgPool -> Dense -> Sigmoid
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model to prevent destroying pretrained weights during initial training
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')   # 0 = real, 1 = fake
    ], name='DeepFakeDetector_MobileNetV2')

    return model


# ─── Training Pipeline ────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  DeepFake Detector — Model Training")
    print("=" * 60)

    # 1. Load data
    X, y = load_dataset()

    # 2. Train / validation split (80 / 20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.astype(int), class_weights_array))
    print(f"[INFO] Class weights: {class_weight_dict}")
    print(f"[INFO] Train: {len(X_train)} frames | Val: {len(X_val)} frames")

    # 3. Build model
    model = build_model()
    model.summary()

    # 4. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 5. Callbacks
    callbacks = [
        # Save best model based on val_accuracy
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 6. Train
    os.makedirs('models', exist_ok=True)
    print("\n[INFO] Starting training ...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # 7. Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULT] Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"[RESULT] Validation Loss     : {val_loss:.4f}")
    print(f"\n[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")
    return history


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train()
