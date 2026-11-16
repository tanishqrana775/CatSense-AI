"""
Calibration & Fusion Optimization Module for MeowMind AI.
Fits temperature scaling parameters (T_vision, T_audio) using Negative Log Likelihood (NLL)
on held-out validation sets and selects optimal fusion weights (w_vision, w_audio).
Writes output to models/calibration.json.
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score

from backend.config import (
    CANONICAL_CLASSES,
    VISION_VAL_PROBS_PATH,
    AUDIO_VAL_PROBS_PATH,
    CALIBRATION_PATH,
    MODELS_DIR
)

def softmax(logits):
    """Numerically stable softmax."""
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

def fit_temperature(probs, targets):
    """
    Fits scalar temperature T > 0 by minimizing Negative Log-Likelihood (NLL)
    using bounded 1-D scalar optimization.
    """
    # Convert probabilities to log-space (pseudo-logits)
    eps = 1e-7
    log_probs = np.log(np.clip(probs, eps, 1.0))
    
    # If targets are one-hot, get indices
    if targets.ndim > 1:
        target_indices = np.argmax(targets, axis=1)
    else:
        target_indices = targets

    def nll_obj(T):
        scaled_logits = log_probs / T
        cal_probs = softmax(scaled_logits)
        # Select target probabilities
        n_samples = len(target_indices)
        correct_probs = cal_probs[np.arange(n_samples), target_indices]
        nll = -np.mean(np.log(np.clip(correct_probs, eps, 1.0)))
        return nll

    res = minimize_scalar(nll_obj, bounds=(0.1, 5.0), method="bounded")
    optimal_T = float(res.x)
    return optimal_T

def main():
    print("=" * 60)
    print("CALIBRATION & FUSION CONFIGURATION OPTIMIZATION")
    print("=" * 60)

    # 1. Calibrate Vision Model
    if VISION_VAL_PROBS_PATH.exists():
        vision_data = np.load(str(VISION_VAL_PROBS_PATH))
        v_probs = vision_data["probabilities"]
        v_targets = vision_data["targets"]
        v_true = np.argmax(v_targets, axis=1) if v_targets.ndim > 1 else v_targets
        v_acc = accuracy_score(v_true, np.argmax(v_probs, axis=1))
        
        T_vision = fit_temperature(v_probs, v_targets)
        print(f"[Vision] Held-out Accuracy: {v_acc*100:.2f}% | Fitted Temperature T_v: {T_vision:.4f}")
    else:
        print(f"Warning: Vision val probs not found at {VISION_VAL_PROBS_PATH}, using default T_vision=1.0")
        v_acc = 0.50
        T_vision = 1.0

    # 2. Calibrate Audio Model
    if AUDIO_VAL_PROBS_PATH.exists():
        audio_data = np.load(str(AUDIO_VAL_PROBS_PATH))
        a_probs = audio_data["probabilities"]
        a_targets = audio_data["targets"]
        a_true = np.argmax(a_targets, axis=1) if a_targets.ndim > 1 else a_targets
        a_acc = accuracy_score(a_true, np.argmax(a_probs, axis=1))
        
        T_audio = fit_temperature(a_probs, a_targets)
        print(f"[Audio] Out-of-Fold Accuracy: {a_acc*100:.2f}% | Fitted Temperature T_a: {T_audio:.4f}")
    else:
        print(f"Warning: Audio val probs not found at {AUDIO_VAL_PROBS_PATH}, using default T_audio=1.0")
        a_acc = 0.25
        T_audio = 1.0

    # 3. Determine Fusion Weights w_vision, w_audio
    # Chance level for 4 balanced classes is 0.25
    CHANCE = 0.25
    v_adv = max(0.01, v_acc - CHANCE)
    a_adv = max(0.01, a_acc - CHANCE)

    raw_w_v = v_adv / (v_adv + a_adv)
    raw_w_a = a_adv / (v_adv + a_adv)

    # If audio is near chance (<= 30%), cap audio weight at 0.20 to protect multimodal accuracy
    if a_acc <= 0.30:
        w_audio = min(0.20, raw_w_a)
        w_vision = 1.0 - w_audio
    else:
        w_vision = round(float(raw_w_v), 3)
        w_audio = round(1.0 - w_vision, 3)

    print(f"\n[Fusion Strategy] Weights: w_vision = {w_vision:.3f}, w_audio = {w_audio:.3f}")

    calibration_config = {
        "T_vision": round(float(T_vision), 4),
        "T_audio": round(float(T_audio), 4),
        "w_vision": round(float(w_vision), 4),
        "w_audio": round(float(w_audio), 4),
        "T_fusion": 1.0,
        "classes": CANONICAL_CLASSES,
        "metrics": {
            "vision_val_acc": round(float(v_acc), 4),
            "audio_cv_acc": round(float(a_acc), 4)
        }
    }

    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration_config, f, indent=4)

    print(f"✅ Calibration configuration written to: {CALIBRATION_PATH}")
    print(json.dumps(calibration_config, indent=2))

if __name__ == "__main__":
    main()
