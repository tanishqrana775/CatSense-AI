"""
Modernized Audio Training Pipeline for MeowMind AI.
Replaces the overparameterized CNN-LSTM with:
- Lightweight 2D-CNN with Depthwise Separable Convolutions & Global Average Pooling (< 40k parameters)
- Strict canonical class indexing ['Angry', 'Fearful', 'Happy', 'Sad']
- Stratified 5-Fold Cross-Validation on original audio files (Split FIRST, then augment!)
- High-performance audio augmentation: SNR-based pink/white noise, pitch shifting, time-stretching, gain
- Graph-native SpecAugment Keras Layer (Frequency and Time masking)
- Out-of-fold calibration probability export
- Realistic, un-fabricated validation metrics
"""
import os
import sys
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    CANONICAL_CLASSES,
    SAMPLE_RATE,
    DURATION,
    TARGET_LEN,
    N_MELS,
    TIME_STEPS,
    AUDIO_MODEL_PATH,
    AUDIO_VAL_PROBS_PATH,
    MODELS_DIR
)
from backend.audio_features import load_audio_waveform, audio_to_logmel

MODELS_DIR.mkdir(exist_ok=True)

# ==========================================================
# 1. GRAPH-NATIVE SPECAUGMENT KERAS LAYER
# ==========================================================
@keras.saving.register_keras_serializable(package="MeowMind")
class SpecAugment(layers.Layer):
    """
    SpecAugment implementation for Time and Frequency Masking.
    Active strictly during training; acts as identity during inference.
    """
    def __init__(self, freq_mask_param=8, time_mask_param=12, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, inputs, training=None):
        if not training:
            return inputs

        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        n_mels = tf.shape(inputs)[2]

        # 1. Frequency masking
        f_len = tf.random.uniform([], minval=1, maxval=self.freq_mask_param + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=n_mels - f_len, dtype=tf.int32)
        mel_indices = tf.range(n_mels, dtype=tf.int32)
        f_mask = tf.logical_and(mel_indices >= f0, mel_indices < f0 + f_len)
        f_mask = tf.reshape(f_mask, (1, 1, n_mels, 1))

        # 2. Time masking
        t_len = tf.random.uniform([], minval=1, maxval=self.time_mask_param + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=time_steps - t_len, dtype=tf.int32)
        time_indices = tf.range(time_steps, dtype=tf.int32)
        t_mask = tf.logical_and(time_indices >= t0, time_indices < t0 + t_len)
        t_mask = tf.reshape(t_mask, (1, time_steps, 1, 1))

        combined_mask = tf.logical_or(f_mask, t_mask)
        # Standardized features have mean 0, so mask to 0.0
        return tf.where(combined_mask, tf.zeros_like(inputs), inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
        })
        return config

def unit_test_specaugment():
    """Confirms SpecAugment modifies tensors in training and is identical in inference."""
    test_layer = SpecAugment(freq_mask_param=10, time_mask_param=15)
    dummy_input = tf.ones((2, TIME_STEPS, N_MELS, 1), dtype=tf.float32)

    eval_out = test_layer(dummy_input, training=False)
    assert tf.reduce_all(eval_out == dummy_input), "SpecAugment failed: modified tensor in inference mode!"

    train_out = test_layer(dummy_input, training=True)
    assert tf.reduce_any(train_out != dummy_input), "SpecAugment failed: no mask applied in training mode!"
    print("✅ SpecAugment layer unit-test passed successfully.")

# ==========================================================
# 2. WAVEFORM AUGMENTATION HELPERS
# ==========================================================
def add_snr_noise(y, min_snr=10, max_snr=25):
    """Injects white Gaussian noise calibrated to random SNR in dB."""
    rms_signal = np.sqrt(np.mean(y**2)) + 1e-9
    snr_db = np.random.uniform(min_snr, max_snr)
    noise = np.random.randn(len(y)).astype(np.float32)
    rms_noise = np.sqrt(np.mean(noise**2)) + 1e-9
    snr_factor = rms_signal / (rms_noise * (10.0 ** (snr_db / 20.0)))
    return (y + noise * snr_factor).astype(np.float32)

def augment_waveform_samples(y, sr):
    """
    Generates 6 distinct acoustic variations for a single training waveform:
    1. Original
    2. SNR Noise (High SNR)
    3. SNR Noise (Low SNR)
    4. Pitch shift up (+1 to +2 semitones)
    5. Pitch shift down (-1 to -2 semitones)
    6. Time stretch / compression (0.9x or 1.1x)
    7. Gain adjustment (+-2.5 dB)
    """
    variants = [y]

    # 1. Noise variants
    variants.append(add_snr_noise(y, 18, 25))
    variants.append(add_snr_noise(y, 10, 18))

    # 2. Pitch shifts
    try:
        variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
        variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    except Exception:
        pass

    # 3. Time stretch
    try:
        stretched = librosa.effects.time_stretch(y, rate=0.92)
        variants.append(stretched)
        compressed = librosa.effects.time_stretch(y, rate=1.08)
        variants.append(compressed)
    except Exception:
        pass

    # 4. Gain perturbation
    gain_db = np.random.uniform(-3.0, 3.0)
    variants.append(y * (10.0 ** (gain_db / 20.0)))

    return variants

# ==========================================================
# 3. LIGHTWEIGHT 2D-CNN MODEL BUILDER
# ==========================================================
def build_lightweight_audio_model(num_classes=4):
    inputs = keras.Input(shape=(TIME_STEPS, N_MELS, 1), name="audio_input")
    x = SpecAugment(freq_mask_param=8, time_mask_param=12)(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2 - Depthwise Separable Convolution
    x = layers.SeparableConv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3 - Depthwise Separable Convolution
    x = layers.SeparableConv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling eliminates Dense parameter explosion
    x = layers.GlobalAveragePooling2D(name="audio_embedding")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="audio_prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CatAudio_Lightweight_2DCNN")
    return model

# ==========================================================
# 4. DATA LOADING & STRATIFIED 5-FOLD CV
# ==========================================================
def main():
    unit_test_specaugment()

    dataset_root = SCRIPT_DIR
    print(f"\nScanning audio dataset at: {dataset_root}")
    
    file_records = []
    for emotion in CANONICAL_CLASSES:
        folder = dataset_root / emotion
        if not folder.exists():
            print(f"Warning: Folder {folder} not found!")
            continue
        for f in os.listdir(folder):
            if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                file_records.append({
                    "path": folder / f,
                    "label": emotion,
                    "target": CANONICAL_CLASSES.index(emotion)
                })

    print(f"Found {len(file_records)} total audio recordings across {len(CANONICAL_CLASSES)} classes.")
    for c_idx, c_name in enumerate(CANONICAL_CLASSES):
        count = sum(1 for r in file_records if r["target"] == c_idx)
        print(f"  - {c_name} (index {c_idx}): {count} files")

    paths = [r["path"] for r in file_records]
    targets = np.array([r["target"] for r in file_records])

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_predictions = np.zeros((len(file_records), len(CANONICAL_CLASSES)), dtype=np.float32)
    fold_accuracies = []
    fold_f1s = []

    print("\n" + "="*60)
    print("STARTING STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("(Split first, augment train fold only!)")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, targets), 1):
        print(f"\n--- FOLD {fold} / 5 ---")
        train_paths = [paths[i] for i in train_idx]
        train_targets = targets[train_idx]

        val_paths = [paths[i] for i in val_idx]
        val_targets = targets[val_idx]

        # 1. Augment training split
        X_train_list, y_train_list = [], []
        for p, t in zip(train_paths, train_targets):
            y_audio, sr = load_audio_waveform(p)
            augmented_waveforms = augment_waveform_samples(y_audio, sr)
            for w in augmented_waveforms:
                feat = audio_to_logmel(w, sr)
                X_train_list.append(feat)
                y_train_list.append(t)

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train_cat = keras.utils.to_categorical(y_train_list, num_classes=len(CANONICAL_CLASSES))

        # 2. Extract validation features (Strictly NO augmentation on validation!)
        X_val_list = []
        for p in val_paths:
            y_audio, sr = load_audio_waveform(p)
            feat = audio_to_logmel(y_audio, sr)
            X_val_list.append(feat)

        X_val = np.array(X_val_list, dtype=np.float32)
        y_val_cat = keras.utils.to_categorical(val_targets, num_classes=len(CANONICAL_CLASSES))

        print(f"Fold {fold}: Train samples={len(X_train)} (from {len(train_idx)} files), Val samples={len(X_val)}")

        # Build & compile fold model
        fold_model = build_lightweight_audio_model(num_classes=len(CANONICAL_CLASSES))
        fold_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy", keras.metrics.F1Score(average="macro", name="f1_macro")]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
        ]

        fold_model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=40,
            batch_size=16,
            verbose=0,
            callbacks=callbacks
        )

        val_preds = fold_model.predict(X_val, verbose=0)
        oof_predictions[val_idx] = val_preds

        fold_acc = accuracy_score(val_targets, np.argmax(val_preds, axis=1))
        fold_f1 = f1_score(val_targets, np.argmax(val_preds, axis=1), average="macro", zero_division=0)
        fold_accuracies.append(fold_acc)
        fold_f1s.append(fold_f1)

        print(f"Fold {fold} Result: Accuracy={fold_acc*100:.2f}%, Macro-F1={fold_f1:.4f}")

    # Out-of-fold Overall Performance
    mean_acc = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)
    oof_preds_labels = np.argmax(oof_predictions, axis=1)

    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"Mean CV Accuracy: {mean_acc*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")
    print(f"Mean CV Macro-F1: {mean_f1:.4f} (+/- {np.std(fold_f1s):.4f})")
    print("\nOOF Classification Report:")
    print(classification_report(targets, oof_preds_labels, target_names=CANONICAL_CLASSES, zero_division=0))
    print("OOF Confusion Matrix:")
    print(confusion_matrix(targets, oof_preds_labels))

    # ==========================================================
    # 5. TRAIN FINAL PRODUCTION MODEL ON FULL DATASET
    # ==========================================================
    print("\n" + "="*60)
    print("TRAINING FINAL PRODUCTION MODEL ON ALL 40 RECORDINGS")
    print("="*60)

    X_all_list, y_all_list = [], []
    for p, t in zip(paths, targets):
        y_audio, sr = load_audio_waveform(p)
        augmented_waveforms = augment_waveform_samples(y_audio, sr)
        for w in augmented_waveforms:
            feat = audio_to_logmel(w, sr)
            X_all_list.append(feat)
            y_all_list.append(t)

    X_all = np.array(X_all_list, dtype=np.float32)
    y_all_cat = keras.utils.to_categorical(y_all_list, num_classes=len(CANONICAL_CLASSES))

    final_model = build_lightweight_audio_model(num_classes=len(CANONICAL_CLASSES))
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    final_model.fit(
        X_all, y_all_cat,
        epochs=35,
        batch_size=16,
        verbose=1
    )

    # Save production model
    final_model.save(str(AUDIO_MODEL_PATH))
    print(f"\n[OK] Final Audio Model saved to: {AUDIO_MODEL_PATH}")

    # Save OOF validation probabilities for calibration
    np.savez(
        str(AUDIO_VAL_PROBS_PATH),
        probabilities=oof_predictions,
        targets=keras.utils.to_categorical(targets, num_classes=len(CANONICAL_CLASSES)),
        classes=np.array(CANONICAL_CLASSES)
    )
    print(f"[OK] Audio OOF probabilities saved to: {AUDIO_VAL_PROBS_PATH}")

if __name__ == "__main__":
    main()
