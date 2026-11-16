import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from keras.models import load_model
from keras.utils import to_categorical

# ============================
# CONFIG
# ============================
MODEL_PATH = "emotion_lstm_model.h5"
TEST_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\audiofiles\CAT_AUDIO_Dataset\output_split_dataset\test"

SAMPLE_RATE = 22050
MAX_PAD_LEN = 130
N_MFCC = 80
EMOTIONS = ["Angry", "Sad", "Happy", "Fearful"]

# ============================
# FEATURE EXTRACTION
# ============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
    return mfcc.T


# ============================
# LOAD TEST DATA
# ============================
X_test = []
y_test = []

for emotion in EMOTIONS:
    emotion_folder = os.path.join(TEST_PATH, emotion)
    
    for file in os.listdir(emotion_folder):
        if file.lower().endswith(".mp3"):
            file_path = os.path.join(emotion_folder, file)
            features = extract_features(file_path)
            X_test.append(features)
            y_test.append(emotion)

X_test = np.array(X_test)
y_test = np.array(y_test)

# ============================
# ENCODE LABELS
# ============================
le = LabelEncoder()
le.fit(EMOTIONS)

y_encoded = le.transform(y_test)
y_categorical = to_categorical(y_encoded)

# ============================
# LOAD MODEL
# ============================
model = load_model(MODEL_PATH)

# ============================
# PREDICTIONS
# ============================
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# ============================
# METRICS
# ============================
accuracy = accuracy_score(y_encoded, y_pred_classes)
f1 = f1_score(y_encoded, y_pred_classes, average="macro")

print("✅ Accuracy:", accuracy)
print("✅ Macro F1 Score:", f1)

print("\nClassification Report:\n")
print(classification_report(y_encoded, y_pred_classes, target_names=EMOTIONS))

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_encoded, y_pred_classes)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
plt.yticks(range(len(EMOTIONS)), EMOTIONS)
plt.colorbar()
plt.show()

# ============================
# PRECISION-RECALL CURVES
# ============================
plt.figure()

for i in range(len(EMOTIONS)):
    precision, recall, _ = precision_recall_curve(
        y_categorical[:, i], 
        y_pred_probs[:, i]
    )
    ap_score = average_precision_score(
        y_categorical[:, i], 
        y_pred_probs[:, i]
    )
    plt.plot(recall, precision, label=f"{EMOTIONS[i]} (AP={ap_score:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()