"""
Production Model Loader for MeowMind AI.
Lazy + Cached model loader: TensorFlow models are loaded on first prediction request,
allowing Gunicorn to boot instantly (<1 second) and stay strictly within Render's 512 MB memory limit.
"""
import os
import json
import logging
from pathlib import Path

# Enforce minimal TF logging and single thread before TF / Keras imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger("catsense.models")

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from backend.config import (
    CANONICAL_CLASSES,
    VISION_MODEL_PATH,
    AUDIO_MODEL_PATH,
    CALIBRATION_PATH
)

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._vision_model = None
            cls._instance._audio_model = None
            cls._instance._calibration = None
            cls._instance.classes = CANONICAL_CLASSES
        return cls._instance

    def load_vision_model(self):
        """Lazy loads and caches the vision model upon first request."""
        if self._vision_model is not None:
            return self._vision_model

        if not VISION_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"[CRITICAL] Vision model not found at {VISION_MODEL_PATH}. "
                "Ensure train_vision_modern.py has completed successfully."
            )
        try:
            import keras
            logger.info(f"[ModelLoader] Lazy-loading Vision model from {VISION_MODEL_PATH}...")
            print(f"[ModelLoader] Lazy-loading Vision model from {VISION_MODEL_PATH}...")
            self._vision_model = keras.models.load_model(str(VISION_MODEL_PATH), compile=False)
            logger.info("✅ [ModelLoader] Vision model loaded successfully.")
            return self._vision_model
        except Exception as e:
            raise RuntimeError(f"[CRITICAL] Failed to load vision model: {e}")

    def load_audio_model(self):
        """Lazy loads and caches the audio model upon first request."""
        if self._audio_model is not None:
            return self._audio_model

        if not AUDIO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"[CRITICAL] Audio model not found at {AUDIO_MODEL_PATH}. "
                "Ensure train_audio_modern.py has completed successfully."
            )
        try:
            import keras
            from data_analysis.datasets.audiofiles.CAT_AUDIO_Dataset.train_audio_modern import SpecAugment
            logger.info(f"[ModelLoader] Lazy-loading Audio model from {AUDIO_MODEL_PATH}...")
            print(f"[ModelLoader] Lazy-loading Audio model from {AUDIO_MODEL_PATH}...")
            self._audio_model = keras.models.load_model(
                str(AUDIO_MODEL_PATH),
                compile=False,
                custom_objects={"SpecAugment": SpecAugment}
            )
            logger.info("✅ [ModelLoader] Audio model loaded successfully.")
            return self._audio_model
        except Exception as e:
            raise RuntimeError(f"[CRITICAL] Failed to load audio model: {e}")

    def load_calibration(self):
        """Loads and caches calibration JSON."""
        if self._calibration is not None:
            return self._calibration

        if CALIBRATION_PATH.exists():
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                self._calibration = json.load(f)
            logger.info(f"✅ [ModelLoader] Calibration config loaded from {CALIBRATION_PATH}")
        else:
            logger.warning(f"⚠️ [ModelLoader] Calibration config not found at {CALIBRATION_PATH}. Using defaults.")
            self._calibration = {
                "T_vision": 1.0,
                "T_audio": 1.0,
                "w_vision": 0.8,
                "w_audio": 0.2,
                "T_fusion": 1.0,
                "classes": CANONICAL_CLASSES
            }
        return self._calibration

    @property
    def vision_model(self):
        return self.load_vision_model()

    @property
    def audio_model(self):
        return self.load_audio_model()

    @property
    def calibration(self):
        return self.load_calibration()

loader = ModelLoader()

def get_vision_model():
    return loader.vision_model

def get_audio_model():
    return loader.audio_model

def get_calibration_config():
    return loader.calibration
