"""
Production Predictor Module for MeowMind AI.
Provides single-modality and multimodal fused prediction pipelines with:
- Strict canonical class ordering: ['Angry', 'Fearful', 'Happy', 'Sad']
- Preprocessing identical to training (guaranteed zero train/serve skew)
- Calibrated temperature scaling per modality
- Log-linear Bayesian fusion (opinion pooling)
- Comprehensive response structures
- Full elimination of fake heuristic fallbacks
"""
import io
import numpy as np
from PIL import Image
import tensorflow as tf

from backend.config import CANONICAL_CLASSES, IMG_SIZE
from backend.audio_features import load_audio_waveform, audio_to_logmel
from backend.inference.model_loader import loader

def softmax(logits):
    """Numerically stable softmax."""
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

def preprocess_image_tensor(image_source):
    """
    Loads and preprocesses image identically to training pipeline:
    - RGB conversion
    - Resize to (224, 224)
    - float32 array in [0, 255] (no extra manual scaling since EfficientNetV2 rescales internally)
    - Batch dimension added: (1, 224, 224, 3)
    """
    if isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, io.BytesIO):
        image_source.seek(0)
        img = Image.open(image_source).convert("RGB")
    elif hasattr(image_source, "read"):
        img = Image.open(image_source).convert("RGB")
    else:
        img = Image.open(str(image_source)).convert("RGB")

    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

def predict_image(image_source, return_probs=False):
    """
    Predicts cat emotion from an image file path, buffer, or bytes.
    Returns:
        If return_probs is False: (emotion_label, confidence_score)
        If return_probs is True:  (emotion_label, confidence_score, calibrated_probabilities)
    """
    if not loader.vision_model:
        raise RuntimeError("Vision model is not loaded.")

    try:
        img_tensor = preprocess_image_tensor(image_source)
        # Raw model predictions: shape (1, 4) in CANONICAL_CLASSES order
        raw_probs = loader.vision_model.predict(img_tensor, verbose=0)[0]

        # Apply temperature calibration
        T_v = loader.calibration.get("T_vision", 1.0)
        eps = 1e-8
        log_probs = np.log(np.clip(raw_probs, eps, 1.0))
        calibrated_probs = softmax(log_probs / T_v)

        top_idx = int(np.argmax(calibrated_probs))
        emotion = CANONICAL_CLASSES[top_idx]
        confidence = round(float(calibrated_probs[top_idx]), 4)

        if return_probs:
            return calibrated_probs, emotion, confidence
        return emotion, confidence

    except Exception as e:
        print(f"[ERROR] predict_image failed: {e}")
        raise

def predict_audio(audio_source, return_probs=False):
    """
    Predicts cat emotion from audio file path, buffer, or bytes.
    Returns:
        If return_probs is False: (emotion_label, confidence_score)
        If return_probs is True:  (emotion_label, confidence_score, calibrated_probabilities)
    """
    if not loader.audio_model:
        raise RuntimeError("Audio model is not loaded.")

    try:
        y, sr = load_audio_waveform(audio_source)
        # audio_to_logmel extracts (130, 64, 1) standardized tensor
        features = audio_to_logmel(y, sr)
        audio_tensor = np.expand_dims(features, axis=0) # (1, 130, 64, 1)

        raw_probs = loader.audio_model.predict(audio_tensor, verbose=0)[0]

        # Apply temperature calibration
        T_a = loader.calibration.get("T_audio", 1.0)
        eps = 1e-8
        log_probs = np.log(np.clip(raw_probs, eps, 1.0))
        calibrated_probs = softmax(log_probs / T_a)

        top_idx = int(np.argmax(calibrated_probs))
        emotion = CANONICAL_CLASSES[top_idx]
        confidence = round(float(calibrated_probs[top_idx]), 4)

        if return_probs:
            return calibrated_probs, emotion, confidence
        return emotion, confidence

    except Exception as e:
        print(f"[ERROR] predict_audio failed: {e}")
        raise

class FusionResult(tuple):
    """
    Dual representation of Fusion output:
    1. As a 3-tuple: (fused_label, fused_confidence, fused_probabilities_array)
    2. As an object/dict with attributes: .emotion, .confidence, .probabilities, .per_modality, .modalities_used
    """
    def __new__(cls, emotion, confidence, probabilities, per_modality=None, modalities_used=None):
        return super().__new__(cls, (emotion, confidence, probabilities))

    def __init__(self, emotion, confidence, probabilities, per_modality=None, modalities_used=None):
        self.emotion = emotion
        self.confidence = confidence
        self.probabilities = probabilities
        self.per_modality = per_modality or {}
        self.modalities_used = modalities_used or []

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "emotion":
                return self.emotion
            elif item == "confidence":
                return self.confidence
            elif item == "probabilities":
                return {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, self.probabilities)}
            elif item == "per_modality":
                return self.per_modality
            elif item == "modalities_used":
                return self.modalities_used
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self):
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, self.probabilities)},
            "per_modality": self.per_modality,
            "modalities_used": self.modalities_used
        }

def fuse(p_v=None, p_a=None):
    """
    Log-linear Bayesian fusion (log-opinion pool):
    P_fused = Softmax( (w_v * log(p_v) + w_a * log(p_a)) / T_fusion )
    
    Handles single-modality gracefully when one modality is None.
    Returns FusionResult tuple: (fused_label, fused_confidence, fused_probabilities_array).
    """
    w_v = loader.calibration.get("w_vision", 0.8)
    w_a = loader.calibration.get("w_audio", 0.2)
    T_fusion = loader.calibration.get("T_fusion", 1.0)
    eps = 1e-8

    modalities_used = []
    per_modality = {}

    if p_v is not None and p_a is not None:
        modalities_used = ["image", "audio"]
        sum_w = w_v + w_a
        w_v_norm = w_v / sum_w
        w_a_norm = w_a / sum_w

        log_v = np.log(np.clip(p_v, eps, 1.0))
        log_a = np.log(np.clip(p_a, eps, 1.0))

        fused_logits = (w_v_norm * log_v + w_a_norm * log_a) / T_fusion
        fused_probs = softmax(fused_logits)

        per_modality["image"] = {
            "emotion": CANONICAL_CLASSES[int(np.argmax(p_v))],
            "confidence": round(float(np.max(p_v)), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_v)}
        }
        per_modality["audio"] = {
            "emotion": CANONICAL_CLASSES[int(np.argmax(p_a))],
            "confidence": round(float(np.max(p_a)), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_a)}
        }

    elif p_v is not None:
        modalities_used = ["image"]
        fused_probs = p_v
        per_modality["image"] = {
            "emotion": CANONICAL_CLASSES[int(np.argmax(p_v))],
            "confidence": round(float(np.max(p_v)), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_v)}
        }

    elif p_a is not None:
        modalities_used = ["audio"]
        fused_probs = p_a
        per_modality["audio"] = {
            "emotion": CANONICAL_CLASSES[int(np.argmax(p_a))],
            "confidence": round(float(np.max(p_a)), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_a)}
        }
    else:
        raise ValueError("Cannot perform fusion: neither visual nor audio modality was provided.")

    top_idx = int(np.argmax(fused_probs))
    fused_emotion = CANONICAL_CLASSES[top_idx]
    fused_confidence = round(float(fused_probs[top_idx]), 4)

    return FusionResult(
        emotion=fused_emotion,
        confidence=fused_confidence,
        probabilities=fused_probs,
        per_modality=per_modality,
        modalities_used=modalities_used
    )
