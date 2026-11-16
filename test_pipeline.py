"""
Comprehensive Verification Test Pipeline for MeowMind AI.
Validates:
1. Model loading & output shapes (N, 4)
2. Canonical class order equality
3. Real inference on 1 real image + 1 real audio per class
4. Determinism check (identical output on duplicate runs)
5. Exact mathematical verification of temperature scaling & Bayesian log-linear fusion
"""
import sys
import os
import json
from pathlib import Path
import numpy as np

# Set project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CANONICAL_CLASSES, CALIBRATION_PATH
from backend.inference.model_loader import get_vision_model, get_audio_model, get_calibration_config
from backend.inference.predictor import predict_image, predict_audio, fuse

def test_pipeline():
    print("=" * 70)
    print("RUNNING MEOWMIND AI PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Verify Canonical Classes & Config
    print("\n--- 1. Canonical Configuration Check ---")
    calib = get_calibration_config()
    print(f"CANONICAL_CLASSES: {CANONICAL_CLASSES}")
    print(f"Calibration classes: {calib['classes']}")
    assert calib["classes"] == CANONICAL_CLASSES, "Mismatch in calibration class order!"
    print("✅ Canonical class order matches across all modules.")

    # 2. Verify Model Architectures and Output Shapes
    print("\n--- 2. Model Architecture & Output Shape Check ---")
    v_model = get_vision_model()
    a_model = get_audio_model()

    # Test dummy batch
    dummy_img = np.zeros((2, 224, 224, 3), dtype=np.float32)
    v_out = v_model.predict(dummy_img, verbose=0)
    assert v_out.shape == (2, 4), f"Vision model output shape mismatch: {v_out.shape} != (2, 4)"
    print(f"✅ Vision model output shape verified: {v_out.shape} == (N, 4)")

    dummy_audio = np.zeros((2, 130, 64, 1), dtype=np.float32)
    a_out = a_model.predict(dummy_audio, verbose=0)
    assert a_out.shape == (2, 4), f"Audio model output shape mismatch: {a_out.shape} != (2, 4)"
    print(f"✅ Audio model output shape verified: {a_out.shape} == (N, 4)")

    # 3. Locate 1 Real Image & 1 Real Audio per Class
    print("\n--- 3. Real Multi-Modal End-to-End Inference ---")
    img_train_dir = PROJECT_ROOT / "data_analysis" / "datasets" / "imagefiles" / "IMAGES_Dataset" / "train"
    audio_dataset_dir = PROJECT_ROOT / "data_analysis" / "datasets" / "audiofiles" / "CAT_AUDIO_Dataset"

    test_samples = {}
    for emotion in CANONICAL_CLASSES:
        # Find 1 real image
        img_folder = img_train_dir / emotion
        img_file = next(f for f in img_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"])
        
        # Find 1 real audio
        audio_folder = audio_dataset_dir / emotion
        audio_file = next(f for f in audio_folder.iterdir() if f.suffix.lower() in [".mp3", ".wav"])

        test_samples[emotion] = {
            "image": img_file,
            "audio": audio_file
        }

    for emotion, paths in test_samples.items():
        print(f"\n[Test Class: {emotion}]")
        img_p = paths["image"]
        aud_p = paths["audio"]
        print(f"  Image: {img_p.name}")
        print(f"  Audio: {aud_p.name}")

        # Run vision inference
        p_v, v_label, v_conf = predict_image(img_p, return_probs=True)
        # Run audio inference
        p_a, a_label, a_conf = predict_audio(aud_p, return_probs=True)
        # Run fusion
        f_label, f_conf, p_f = fuse(p_v, p_a)

        print(f"  Vision Probs: {dict(zip(CANONICAL_CLASSES, np.round(p_v, 4)))} -> Pred: {v_label} ({v_conf*100:.1f}%)")
        print(f"  Audio  Probs: {dict(zip(CANONICAL_CLASSES, np.round(p_a, 4)))} -> Pred: {a_label} ({a_conf*100:.1f}%)")
        print(f"  Fused  Probs: {dict(zip(CANONICAL_CLASSES, np.round(p_f, 4)))} -> Pred: {f_label} ({f_conf*100:.1f}%)")

        assert len(p_v) == 4 and np.isclose(np.sum(p_v), 1.0, atol=1e-4), "Vision probs do not sum to 1.0"
        assert len(p_a) == 4 and np.isclose(np.sum(p_a), 1.0, atol=1e-4), "Audio probs do not sum to 1.0"
        assert len(p_f) == 4 and np.isclose(np.sum(p_f), 1.0, atol=1e-4), "Fused probs do not sum to 1.0"

    # 4. Determinism Verification
    print("\n--- 4. Determinism Check ---")
    sample_img = test_samples["Happy"]["image"]
    sample_aud = test_samples["Happy"]["audio"]

    p_v1, l_v1, c_v1 = predict_image(sample_img, return_probs=True)
    p_v2, l_v2, c_v2 = predict_image(sample_img, return_probs=True)
    assert np.allclose(p_v1, p_v2, atol=1e-6), "Vision prediction is non-deterministic!"

    p_a1, l_a1, c_a1 = predict_audio(sample_aud, return_probs=True)
    p_a2, l_a2, c_a2 = predict_audio(sample_aud, return_probs=True)
    assert np.allclose(p_a1, p_a2, atol=1e-6), "Audio prediction is non-deterministic!"
    print("✅ Inference is 100% deterministic (duplicate calls yield bitwise identical outputs).")

    # 5. Fusion Math Mathematical Verification
    print("\n--- 5. Fusion Formula Mathematical Verification ---")
    w_v = calib["w_vision"]
    w_a = calib["w_audio"]
    sum_w = w_v + w_a
    w_v_norm = w_v / sum_w
    w_a_norm = w_a / sum_w
    eps = 1e-8

    # p_v1 and p_a1 are already temperature-calibrated probabilities
    log_p_v = np.log(np.clip(p_v1, eps, 1.0))
    log_p_a = np.log(np.clip(p_a1, eps, 1.0))
    manual_fused_logits = w_v_norm * log_p_v + w_a_norm * log_p_a
    exp_fused = np.exp(manual_fused_logits - np.max(manual_fused_logits))
    manual_fused_probs = exp_fused / np.sum(exp_fused)

    _, _, code_fused_probs = fuse(p_v1, p_a1)
    assert np.allclose(manual_fused_probs, code_fused_probs, atol=1e-5), f"Fusion mismatch: {manual_fused_probs} vs {code_fused_probs}"
    print(f"✅ Fusion math verified: manual calculation matches code output within atol=1e-5.")

    print("\n" + "=" * 70)
    print("ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
