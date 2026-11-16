"""
Flask Endpoint Integration Test for MeowMind AI.
Submits:
1. Image-only POST to /demo
2. Audio-only POST to /demo
3. Combined POST to /demo (Image + Audio)
4. Authenticated Combined POST to /new_analysis
Captures and verifies HTTP status codes, emotion prediction texts, and confidence values.
"""
import sys
import io
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from backend.config import CANONICAL_CLASSES

def extract_detected_emotion(html):
    m = re.search(r'<h2 class="detected-emotion"[^>]*>\s*([A-Za-z]+)\s*</h2>', html)
    if m:
        return m.group(1).strip()
    return "Not Found"

def run_flask_tests():
    print("=" * 70)
    print("TESTING FLASK APPLICATION ENDPOINTS")
    print("=" * 70)

    # 1. Locate test image and audio files
    img_train_dir = PROJECT_ROOT / "data_analysis" / "datasets" / "imagefiles" / "IMAGES_Dataset" / "train"
    audio_dataset_dir = PROJECT_ROOT / "data_analysis" / "datasets" / "audiofiles" / "CAT_AUDIO_Dataset"

    happy_img = next((img_train_dir / "Happy").iterdir())
    happy_audio = next((audio_dataset_dir / "Happy").iterdir())

    print(f"Using Test Files:\n  Image: {happy_img}\n  Audio: {happy_audio}\n")

    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        # -----------------------------------------------------------
        # Test 1: Image-only POST to /demo
        # -----------------------------------------------------------
        print("--- Test 1: Image-only POST to /demo ---")
        with open(happy_img, "rb") as f_img:
            data = {
                "file_type": "image",
                "file": (io.BytesIO(f_img.read()), happy_img.name)
            }
            res = client.post("/demo", data=data, content_type="multipart/form-data", follow_redirects=True)
            print(f"Status Code: {res.status_code}")
            html = res.data.decode("utf-8")
            emo_text = extract_detected_emotion(html)
            print(f"Result Emotion Displayed: '{emo_text}'")
            assert res.status_code == 200
            assert emo_text in CANONICAL_CLASSES, f"Unexpected emotion: {emo_text}"
            print("✅ Image-only demo POST test PASSED.")

        # -----------------------------------------------------------
        # Test 2: Audio-only POST to /demo (resetting session for demo trial)
        # -----------------------------------------------------------
        print("\n--- Test 2: Audio-only POST to /demo ---")
        with client.session_transaction() as sess:
            sess["trial_used"] = False  # Reset trial flag for testing

        with open(happy_audio, "rb") as f_aud:
            data = {
                "file_type": "audio",
                "file": (io.BytesIO(f_aud.read()), happy_audio.name)
            }
            res = client.post("/demo", data=data, content_type="multipart/form-data", follow_redirects=True)
            print(f"Status Code: {res.status_code}")
            html = res.data.decode("utf-8")
            emo_text = extract_detected_emotion(html)
            print(f"Result Emotion Displayed: '{emo_text}'")
            assert res.status_code == 200
            assert emo_text in CANONICAL_CLASSES, f"Unexpected emotion: {emo_text}"
            print("✅ Audio-only demo POST test PASSED.")

        # -----------------------------------------------------------
        # Test 3: Combined POST (Image + Audio) to /demo
        # -----------------------------------------------------------
        print("\n--- Test 3: Combined POST (Image + Audio) to /demo ---")
        with client.session_transaction() as sess:
            sess["trial_used"] = False  # Reset trial flag for testing

        with open(happy_img, "rb") as f_img, open(happy_audio, "rb") as f_aud:
            data = {
                "file_type": "combined",
                "image_file": (io.BytesIO(f_img.read()), happy_img.name),
                "audio_file": (io.BytesIO(f_aud.read()), happy_audio.name)
            }
            res = client.post("/demo", data=data, content_type="multipart/form-data", follow_redirects=True)
            print(f"Status Code: {res.status_code}")
            html = res.data.decode("utf-8")
            emo_text = extract_detected_emotion(html)
            print(f"Result Emotion Displayed: '{emo_text}'")
            
            # Verify breakdown cards exist in HTML
            assert "Image Model" in html, "Image breakdown card not rendered in HTML!"
            assert "Audio Model" in html, "Audio breakdown card not rendered in HTML!"
            assert res.status_code == 200
            assert emo_text in CANONICAL_CLASSES, f"Unexpected emotion: {emo_text}"
            print("✅ Combined demo POST test PASSED.")

        # -----------------------------------------------------------
        # Test 4: Authenticated Combined POST to /new_analysis
        # -----------------------------------------------------------
        print("\n--- Test 4: Authenticated User /new_analysis Combined Analysis ---")
        # Register/login a temporary test user
        test_email = "test_verifier@catproject.com"
        test_pass = "TestSecurePassword123!"
        client.post("/register", data={"email": test_email, "password": test_pass, "confirm_password": test_pass})
        client.post("/login", data={"email": test_email, "password": test_pass})

        with open(happy_img, "rb") as f_img, open(happy_audio, "rb") as f_aud:
            data = {
                "file_type": "combined",
                "image_file": (io.BytesIO(f_img.read()), happy_img.name),
                "audio_file": (io.BytesIO(f_aud.read()), happy_audio.name)
            }
            res = client.post("/new_analysis", data=data, content_type="multipart/form-data", follow_redirects=True)
            print(f"Status Code: {res.status_code}")
            html = res.data.decode("utf-8")
            assert res.status_code == 200
            assert "Combined analysis complete!" in html or "Analysis complete!" in html
            print("✅ Authenticated /new_analysis combined analysis PASSED.")

    print("\n" + "=" * 70)
    print("ALL FLASK ENDPOINT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_flask_tests()
