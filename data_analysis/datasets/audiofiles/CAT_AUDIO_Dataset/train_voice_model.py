import os
import numpy as np
import librosa
import keras
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D, Input

# ============================
# CONFIG
# ============================
TRAIN_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\audiofiles\CAT_AUDIO_Dataset\output_split_dataset\train"
TEST_PATH = r"C:\Users\tanis\Downloads\Cat_emotion_classification_Project\data_analysis\datasets\audiofiles\CAT_AUDIO_Dataset\output_split_dataset\test"

SAMPLE_RATE = 22050
MAX_PAD_LEN = 130   # Time steps
N_MFCC = 80         # Features
EMOTIONS = ["Angry", "Sad", "Happy", "Fearful"]

# ============================
# AUDIO AUGMENTATION
# ============================
def augment_audio(y, sr):
    # Add white noise
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise

    # Pitch shift
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    return y_noise, y_pitch


# ============================
# FEATURE EXTRACTION
# ============================
def extract_features(file_path, augment=False):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    features = []

    # Original
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
    features.append(mfcc.T)

    if augment:
        y_noise, y_pitch = augment_audio(y, sr)

        for aug_y in [y_noise, y_pitch]:
            mfcc = librosa.feature.mfcc(y=aug_y, sr=sr, n_mfcc=N_MFCC)
            mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
            features.append(mfcc.T)

    return features


# ============================
# LOAD DATASET
# ============================
def load_data(dataset_path, augment=False):
    X = []
    y = []

    for emotion in EMOTIONS:
        emotion_folder = os.path.join(dataset_path, emotion)

        for file in tqdm(os.listdir(emotion_folder)):
            file_path = os.path.join(emotion_folder, file)

            if file.lower().endswith((".wav", ".mp3", ".flac")):
                features = extract_features(file_path, augment)

                for feature in features:
                    X.append(feature)
                    y.append(emotion)

    return np.array(X), np.array(y)


print("Loading training data...")
X_train, y_train = load_data(TRAIN_PATH, augment=True)

print("Loading testing data...")
X_test, y_test = load_data(TEST_PATH, augment=False)

# ============================
# ENCODE LABELS
# ============================
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

y_train_cat = to_categorical(y_train_encoded)
y_test_cat = to_categorical(y_test_encoded)

num_classes = y_train_cat.shape[1]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ============================
# BUILD MODEL
# ============================
model = Sequential([
    Input(shape=(MAX_PAD_LEN, N_MFCC)),

    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=4),

    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),

    LSTM(64),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy',
        keras.metrics.F1Score(average='macro', name='f1_score')
    ]
)

model.summary()

# ============================
# TRAIN MODEL
# ============================
history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=30,
    batch_size=32
)

# ============================
# SAVE MODEL
# ============================
model.save("emotion_lstm_model.h5")

print("✅ Training complete and model saved!")