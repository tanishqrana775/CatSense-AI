import os
import shutil
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURATION
# Defined identically to backend/config.py
# =========================
CANONICAL_CLASSES = ["Angry", "Fearful", "Happy", "Sad"]
EMOTIONS = CANONICAL_CLASSES

DATASET_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\audiofiles\CAT_AUDIO_Dataset"
OUTPUT_PATH = os.path.join(DATASET_PATH, "output_split_dataset")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for split in ["train", "test"]:
    for emotion in EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_PATH, split, emotion), exist_ok=True)

# =========================
# COLLECT FILE PATHS & LABELS
# =========================
file_paths = []
labels = []

for emotion in EMOTIONS:
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(emotion_folder):
        continue
    for file in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            file_paths.append(file_path)
            labels.append(emotion)

# =========================
# TRAIN-TEST SPLIT (Stratified)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    file_paths,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

# =========================
# COPY FILES TO TRAIN FOLDER
# =========================
for file_path, label in zip(X_train, y_train):
    destination = os.path.join(OUTPUT_PATH, "train", label)
    shutil.copy(file_path, destination)

# =========================
# COPY FILES TO TEST FOLDER
# =========================
for file_path, label in zip(X_test, y_test):
    destination = os.path.join(OUTPUT_PATH, "test", label)
    shutil.copy(file_path, destination)

print("✅ Dataset successfully split!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
