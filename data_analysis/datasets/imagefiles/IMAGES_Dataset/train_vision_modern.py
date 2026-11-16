"""
Modernized Vision Training Pipeline for MeowMind AI.
Uses EfficientNetV2-B0 transfer learning with:
- Strict canonical class indexing ['Angry', 'Fearful', 'Happy', 'Sad']
- Freeze-safe fine-tuning (preserving pre-trained BatchNorm statistics)
- Advanced visual augmentations (RandomFlip, Rotation, Zoom, Translation, Brightness, Contrast, RandomErasing)
- Categorical Focal Loss with label smoothing
- Two-stage training (frozen warmup -> selective non-BN fine-tuning)
- Generates validation probabilities for temperature calibration
"""
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    CANONICAL_CLASSES,
    IMG_SIZE,
    VISION_MODEL_PATH,
    VISION_VAL_PROBS_PATH,
    MODELS_DIR
)

MODELS_DIR.mkdir(exist_ok=True)

# 1. Dataset Configuration
TRAIN_DIR = SCRIPT_DIR / "train"
VAL_DIR = SCRIPT_DIR / "val"
BATCH_SIZE = 16

print(f"Loading datasets from {TRAIN_DIR} and {VAL_DIR}")
print(f"Using Canonical Classes: {CANONICAL_CLASSES}")

train_ds = keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="categorical",
    class_names=CANONICAL_CLASSES, # Strict explicit class ordering
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    str(VAL_DIR),
    labels="inferred",
    label_mode="categorical",
    class_names=CANONICAL_CLASSES, # Strict explicit class ordering
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Compute class weights from training set to handle minor imbalances
train_labels = []
for _, y_batch in train_ds:
    train_labels.extend(np.argmax(y_batch.numpy(), axis=1))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CANONICAL_CLASSES)),
    y=train_labels
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("Computed Class Weights:", class_weight_dict)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 2. Data Augmentation Pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
    layers.RandomContrast(0.2),
    layers.RandomErasing(scale=(0.02, 0.2), value_range=(0, 255))
], name="vision_augmentation")

# 3. Model Architecture
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_input")
augmented = data_augmentation(inputs)

# Base model: EfficientNetV2-B0 expects raw [0, 255] pixels and rescales internally
base_model = keras.applications.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Initially freeze the entire base model
base_model.trainable = False

# training=False preserves pre-trained BatchNormalization statistics!
x = base_model(augmented, training=False)
x = layers.GlobalAveragePooling2D(name="vision_embedding")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CANONICAL_CLASSES), activation="softmax", name="vision_prediction")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="MeowMind_Vision_EfficientNetV2")

# 4. Focal Loss formulation
focal_loss = keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    label_smoothing=0.1
)

metrics = [
    "accuracy",
    keras.metrics.F1Score(average="macro", name="f1_macro")
]

# ==========================================================
# PHASE A: Head Training (Base Model Frozen)
# ==========================================================
print("\n" + "="*60)
print("PHASE A: Training Classifier Head (Base Model Frozen)")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=metrics
)

phase_a_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight_dict
)

# ==========================================================
# PHASE B: Selective Fine-Tuning (Top ~25 Non-BN Layers)
# ==========================================================
print("\n" + "="*60)
print("PHASE B: Fine-Tuning Top Non-BN Layers")
print("="*60)

base_model.trainable = True

# Keep all BatchNormalization layers frozen; unfreeze only top non-BN layers
non_bn_layers = [l for l in base_model.layers if not isinstance(l, (layers.BatchNormalization, keras.layers.BatchNormalization))]
unfreeze_count = min(25, len(non_bn_layers))
frozen_layers = non_bn_layers[:-unfreeze_count]
trainable_layers = non_bn_layers[-unfreeze_count:]

for l in base_model.layers:
    if isinstance(l, (layers.BatchNormalization, keras.layers.BatchNormalization)):
        l.trainable = False
    elif l in trainable_layers:
        l.trainable = True
    else:
        l.trainable = False

trainable_count = sum(len(l.trainable_weights) for l in model.layers)
print(f"Fine-tuning with top {unfreeze_count} non-BN layers active.")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss,
    metrics=metrics
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        str(VISION_MODEL_PATH),
        monitor="val_f1_macro",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

phase_b_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Ensure the best model is saved to destination
model.save(str(VISION_MODEL_PATH))
print(f"\n[OK] Best vision model saved to: {VISION_MODEL_PATH}")

# ==========================================================
# EVALUATION & CALIBRATION DATA EXPORT
# ==========================================================
print("\n" + "="*60)
print("VALIDATION SET EVALUATION")
print("="*60)

val_probs = []
val_targets = []

for x_val, y_val in val_ds:
    preds = model.predict(x_val, verbose=0)
    val_probs.append(preds)
    val_targets.append(y_val.numpy())

val_probs = np.concatenate(val_probs, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

y_true_indices = np.argmax(val_targets, axis=1)
y_pred_indices = np.argmax(val_probs, axis=1)

print("\nClassification Report (Validation Set):")
print(classification_report(y_true_indices, y_pred_indices, target_names=CANONICAL_CLASSES, zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_true_indices, y_pred_indices)
print(cm)

# Save probabilities and labels for temperature scaling calibration
np.savez(
    str(VISION_VAL_PROBS_PATH),
    probabilities=val_probs,
    targets=val_targets,
    classes=np.array(CANONICAL_CLASSES)
)
print(f"[OK] Validation probabilities saved to: {VISION_VAL_PROBS_PATH}")
