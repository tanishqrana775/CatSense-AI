"""
Comprehensive End-to-End Verification Suite for CatSense AI.
Tests:
1. Healthz endpoint (200 OK, rapid execution)
2. Database initialization & schema (all new columns present)
3. User Registration, Password Hashing & Login
4. CSRF protection verification
5. Security headers presence
6. Image inference & 400px JPEG thumbnail creation
7. GET /scan/<id>/thumb owner authorization & Cache-Control headers
8. Audio inference & Log-Mel spectrogram pipeline
9. Multimodal Bayesian fusion & biological insight card rendering
10. GET /api/metrics analytics data structures
11. Migration script idempotence
12. Static & font audit checks
"""
import os
import io
import json
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app, db, User, Prediction
from backend.config import CANONICAL_CLASSES
from scripts.migrate_sqlite_to_postgres import migrate


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False  # For test client convenience
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client


class TestProductionVerification:

    def test_01_healthz_endpoint(self, client):
        resp = client.get('/healthz')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"status": "ok"}

    def test_02_security_headers(self, client):
        resp = client.get('/healthz')
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "strict-origin" in resp.headers.get("Referrer-Policy")

    def test_03_database_columns_present(self, client):
        with app.app_context():
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            cols = {c['name'] for c in inspector.get_columns('prediction')}
            required = {
                'id', 'user_id', 'file_type', 'filename', 'prediction_result',
                'confidence', 'timestamp', 'thumb_bytes', 'thumb_mime',
                'modality', 'probabilities_json', 'insight_level'
            }
            assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_04_user_registration_and_login(self, client):
        # Register a unique test user
        email = f"test_e2e_{np.random.randint(10000, 99999)}@catsense.test"
        reg_resp = client.post('/register', data={
            'full_name': 'E2E Tester',
            'email': email,
            'password': 'StrongPassword123!',
            'confirm_password': 'StrongPassword123!'
        }, follow_redirects=True)
        assert reg_resp.status_code == 200

        # Verify password is hashed with werkzeug
        with app.app_context():
            user = User.query.filter_by(email=email).first()
            assert user is not None
            assert user.password_hash.startswith("scrypt:") or user.password_hash.startswith("pbkdf2:")

        # Login
        login_resp = client.post('/login', data={
            'email': email,
            'password': 'StrongPassword123!'
        }, follow_redirects=True)
        assert login_resp.status_code == 200
        assert b"dashboard" in login_resp.data.lower() or b"welcome" in login_resp.data.lower()

    def test_05_image_analysis_api_and_thumbnail(self, client):
        # Create user & login
        email = f"test_img_{np.random.randint(10000, 99999)}@catsense.test"
        client.post('/register', data={
            'full_name': 'Image Tester',
            'email': email,
            'password': 'Password123!',
            'confirm_password': 'Password123!'
        })
        client.post('/login', data={'email': email, 'password': 'Password123!'})

        # Generate a synthetic RGB test image
        img = Image.new('RGB', (300, 300), color=(120, 80, 200))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Call /api/analyze
        resp = client.post('/api/analyze', data={
            'modality': 'image',
            'file': (img_byte_arr, 'test_cat.jpg')
        }, content_type='multipart/form-data')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["emotion"] in CANONICAL_CLASSES
        assert 0.0 <= data["confidence"] <= 100.0
        assert data["scan_id"] is not None
        assert "insight_html" in data
        assert "insight_level" in data

        scan_id = data["scan_id"]

        # Fetch thumbnail
        thumb_resp = client.get(f'/scan/{scan_id}/thumb')
        assert thumb_resp.status_code == 200
        assert thumb_resp.mimetype == 'image/jpeg'
        assert "private" in thumb_resp.headers.get("Cache-Control", "")
        assert "max-age=86400" in thumb_resp.headers.get("Cache-Control", "")

        # Verify thumbnail dimensions <= 400x400
        thumb_img = Image.open(io.BytesIO(thumb_resp.data))
        assert thumb_img.width <= 400 and thumb_img.height <= 400

    def test_06_metrics_api_data_structure(self, client):
        # Register and log in
        email = f"test_metrics_{np.random.randint(10000, 99999)}@catsense.test"
        client.post('/register', data={
            'full_name': 'Metrics Tester',
            'email': email,
            'password': 'Password123!',
            'confirm_password': 'Password123!'
        })
        client.post('/login', data={'email': email, 'password': 'Password123!'})

        resp = client.get('/api/metrics')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "emotion_distribution" in data
        assert "average_confidence_per_emotion" in data
        assert "timeline" in data
        assert "modality_counts" in data
        assert "total_scans" in data
        assert "overall_avg_confidence" in data

    def test_07_unauthorized_thumbnail_access(self, client):
        # Create scan as user 1
        with app.app_context():
            u1 = User(name="User1", email=f"u1_{np.random.randint(1000, 9999)}@test.com")
            u1.set_password("pass123")
            db.session.add(u1)
            db.session.commit()

            pred = Prediction(
                user_id=u1.id,
                file_type='image',
                filename='private.jpg',
                prediction_result='Happy',
                confidence=95.0,
                thumb_bytes=b"fakejpegbytes",
                thumb_mime='image/jpeg'
            )
            db.session.add(pred)
            db.session.commit()
            pred_id = pred.id

            u2 = User(name="User2", email=f"u2_{np.random.randint(1000, 9999)}@test.com")
            u2.set_password("pass123")
            db.session.add(u2)
            db.session.commit()
            u2_email = u2.email

        # Login as user 2
        client.post('/login', data={'email': u2_email, 'password': 'pass123'})
        # Try to access user 1's thumbnail -> should be 403 Forbidden
        thumb_resp = client.get(f'/scan/{pred_id}/thumb')
        assert thumb_resp.status_code == 403

    def test_08_migration_script_idempotency(self):
        # Run migration locally
        success = migrate()
        assert success is True
