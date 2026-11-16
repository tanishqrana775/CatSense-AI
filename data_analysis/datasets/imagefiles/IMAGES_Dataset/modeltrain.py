import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -----------------------
# Config
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # ⬆ Increased from 8 (better gradient stability)
EPOCHS = 30

TRAIN_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\imagefiles\IMAGES_Dataset\train"
VAL_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\imagefiles\IMAGES_Dataset\val"

# -----------------------
# Load Train Dataset
# -----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# -----------------------
# Load Validation Dataset
# -----------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------
# Get class info (FIXED missing num_classes)
# -----------------------
class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# -----------------------
# Performance optimization
# -----------------------
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# -----------------------
# Compute Class Weights (NEW - Handles class imbalance)
# -----------------------
labels = []

for _, y in train_ds:
    labels.extend(y.numpy())

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weight_dict = dict(enumerate(class_weights))

print("Class Weights:", class_weight_dict)

# -----------------------
# Data Augmentation (NEW - improves generalization)
# -----------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# -----------------------
# ResNet50 Base Model
# -----------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# -----------------------
# Fine-Tuning Strategy (IMPROVED)
# Freeze early layers, train last 30
# -----------------------
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

# -----------------------
# Build Classifier Head (IMPROVED)
# -----------------------
inputs = keras.Input(shape=(224, 224, 3))

# 🔥 Apply augmentation BEFORE preprocessing
x = data_augmentation(inputs)

# Preprocess for ResNet
x = preprocess_input(x)

# Forward pass through base model
x = base_model(x, training=True)  # training=True for fine-tuning

# Global pooling
x = layers.GlobalAveragePooling2D()(x)

# 🔥 Stronger classifier head
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)   # NEW - stabilizes learning
x = layers.Dropout(0.5)(x)           # Increased dropout

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# -----------------------
# Compile (LOWER LR for fine-tuning)
# -----------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # ⬇ Lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# Callbacks
# -----------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# -----------------------
# Train
# -----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict   # NEW - helps minority classes
)

# -----------------------
# Save Final Model
# -----------------------
model.save("resnet50_cpu_model.keras")
print("Saved → resnet50_cpu_model.keras")