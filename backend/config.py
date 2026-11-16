"""
Single source of truth configuration for MeowMind AI.
Defines canonical class ordering, audio/vision hyperparameters, and artifact paths.
"""
from pathlib import Path

# Canonical Class Order (Used everywhere across vision, audio, fusion, and frontend)
CANONICAL_CLASSES = ["Angry", "Fearful", "Happy", "Sad"]
NUM_CLASSES = len(CANONICAL_CLASSES)
CLASS_TO_INDEX = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASSES)}
INDEX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CANONICAL_CLASSES)}

# Vision Constants
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

# Audio Constants
SAMPLE_RATE = 22050
DURATION = 3.0
TARGET_LEN = int(SAMPLE_RATE * DURATION) # 66150 audio samples
N_MELS = 64
HOP_LENGTH = 512
TIME_STEPS = 130  # np.ceil(66150 / 512) = 130 frames

# Project and Model Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

VISION_MODEL_PATH = MODELS_DIR / "vision_model.keras"
AUDIO_MODEL_PATH = MODELS_DIR / "audio_model.keras"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
VISION_VAL_PROBS_PATH = MODELS_DIR / "vision_val_probs.npz"
AUDIO_VAL_PROBS_PATH = MODELS_DIR / "audio_val_probs.npz"
