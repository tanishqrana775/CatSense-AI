import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\imagefiles\IMAGES_Dataset\resnet50_cpu_model.keras"
TEST_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\imagefiles\IMAGES_Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# =========================
# LOAD TEST DATASET
# =========================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  # IMPORTANT for correct evaluation
)

class_names = test_ds.class_names
n_classes = len(class_names)
print("Test Classes:", class_names)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# =========================
# LOAD MODEL
# =========================
model = keras.models.load_model(MODEL_PATH)

# =========================
# BASIC EVALUATION
# =========================
loss, accuracy = model.evaluate(test_ds, verbose=1)

# =========================
# GET PREDICTIONS
# =========================
y_true = []
y_probs = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_probs.extend(predictions)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)
y_pred = np.argmax(y_probs, axis=1)

# =========================
# METRICS
# =========================
accuracy_calc = accuracy_score(y_true, y_pred)
precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
f1_macro = f1_score(y_true, y_pred, average='macro')

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(n_classes), class_names, rotation=45)
plt.yticks(range(n_classes), class_names)
plt.colorbar()
plt.show()

# =========================
# PRECISION-RECALL CURVES
# =========================
y_true_bin = label_binarize(y_true, classes=range(n_classes))

plt.figure()

for i in range(n_classes):
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true_bin[:, i], y_probs[:, i]
    )

    ap_score = average_precision_score(
        y_true_bin[:, i], y_probs[:, i]
    )

    plt.plot(recall_curve, precision_curve,
             label=f"{class_names[i]} (AP={ap_score:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()

# =========================
# FINAL METRIC SUMMARY
# =========================
print("\n==============================")
print(f"Test Accuracy (Keras)  : {accuracy*100:.2f}%")
print(f"Test Accuracy (Manual) : {accuracy_calc*100:.2f}%")
print(f"Test Loss              : {loss:.4f}")
print(f"Precision (Weighted)   : {precision_weighted:.4f}")
print(f"Recall (Weighted)      : {recall_weighted:.4f}")
print(f"F1 Score (Weighted)    : {f1_weighted:.4f}")
print(f"F1 Score (Macro)       : {f1_macro:.4f}")
print("==============================")