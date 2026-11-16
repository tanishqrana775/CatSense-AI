import os
import io
import json
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from PIL import Image

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, session, jsonify, send_file, make_response
)
from flask_login import (
    LoginManager, login_user, login_required, logout_user, current_user
)
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Import local configuration and modules
from config import Config
from backend.config import CANONICAL_CLASSES
from database.database_logic import db, init_db, User, Prediction, Scan
from backend.inference.predictor import predict_image, predict_audio, fuse
from backend.insights import get_insight

# ==========================================
# APP INITIALIZATION & CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=Config.TEMPLATE_FOLDER,
    static_folder=Config.STATIC_FOLDER
)
app.config.from_object(Config)

# Initialize Security & Database Extensions
init_db(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

ALLOWED_IMAGE_EXTS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_AUDIO_EXTS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTS | ALLOWED_AUDIO_EXTS

EMOJI_MAP = {
    "Angry": "😾",
    "Fearful": "🙀",
    "Happy": "😸",
    "Sad": "😿"
}

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def allowed_file(filename, allowed_set=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

# ==========================================
# IN-MEMORY UPLOAD & THUMBNAIL HELPERS
# ==========================================
def extract_and_validate_image(file_storage):
    """
    Validates uploaded image purely in-memory with Pillow.
    Generates a 400px JPEG thumbnail stored in bytes.
    Returns (raw_bytes, thumb_bytes, safe_filename).
    """
    filename = secure_filename(file_storage.filename or 'image.jpg')
    raw_bytes = file_storage.read()
    if not raw_bytes:
        raise ValueError("Image file is empty.")

    # Strict content verification
    try:
        verifier = Image.open(io.BytesIO(raw_bytes))
        verifier.verify()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file: {e}")

    # Reopen to generate 400px thumbnail
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    thumb = img.copy()
    thumb.thumbnail((400, 400), Image.Resampling.LANCZOS)
    thumb_buffer = io.BytesIO()
    thumb.save(thumb_buffer, format='JPEG', quality=80)
    thumb_bytes = thumb_buffer.getvalue()

    return raw_bytes, thumb_bytes, filename


def extract_and_validate_audio(file_storage):
    """
    Validates audio file and returns (audio_bytes, safe_filename).
    """
    filename = secure_filename(file_storage.filename or 'audio.wav')
    audio_bytes = file_storage.read()
    if not audio_bytes:
        raise ValueError("Audio file is empty.")

    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_AUDIO_EXTS:
        raise ValueError("Unsupported audio format. Supported: WAV, MP3, OGG, FLAC.")

    return audio_bytes, filename


# ==========================================
# USER STATS HELPER
# ==========================================
def get_user_stats(user_id):
    """Computes comprehensive prediction statistics for an authenticated user."""
    all_preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).all()
    total = len(all_preds)

    image_count = sum(1 for p in all_preds if (p.modality == 'image' or p.file_type in ('image', 'Facial')))
    audio_count = sum(1 for p in all_preds if (p.modality == 'audio' or p.file_type in ('audio', 'Vocal')))
    combined_count = sum(1 for p in all_preds if (p.modality in ('both', 'combined') or p.file_type == 'combined'))

    # Canonical emotion distribution
    emotion_counts = {cls: 0 for cls in CANONICAL_CLASSES}
    for p in all_preds:
        res = p.prediction_result.capitalize() if p.prediction_result else "Happy"
        if res in emotion_counts:
            emotion_counts[res] += 1
        else:
            emotion_counts[res] = 1

    recent = all_preds[:10]

    valid_confidences = [p.display_confidence for p in all_preds if p.display_confidence is not None]
    overall_avg_confidence = round(sum(valid_confidences) / len(valid_confidences), 1) if valid_confidences else 0.0

    image_confs = [
        p.display_confidence for p in all_preds
        if (p.modality == 'image' or p.file_type in ('image', 'Facial')) and p.display_confidence is not None
    ]
    audio_confs = [
        p.display_confidence for p in all_preds
        if (p.modality == 'audio' or p.file_type in ('audio', 'Vocal')) and p.display_confidence is not None
    ]

    avg_image_confidence = round(sum(image_confs) / len(image_confs), 1) if image_confs else 0.0
    avg_audio_confidence = round(sum(audio_confs) / len(audio_confs), 1) if audio_confs else 0.0

    most_common = max(emotion_counts.items(), key=lambda x: x[1]) if total > 0 else (None, 0)
    most_common_emotion = most_common if most_common[1] > 0 else None

    return {
        'total_predictions': total,
        'image_count': image_count,
        'audio_count': audio_count,
        'combined_count': combined_count,
        'unique_emotions': sum(1 for k, v in emotion_counts.items() if v > 0),
        'emotion_distribution': emotion_counts,
        'recent_predictions': recent,
        'all_predictions': all_preds,
        'overall_avg_confidence': overall_avg_confidence,
        'avg_image_confidence': avg_image_confidence,
        'avg_audio_confidence': avg_audio_confidence,
        'most_common_emotion': most_common_emotion
    }


# ==========================================
# ROUTES: HEALTH & STATIC
# ==========================================
@app.route('/healthz')
def healthz():
    """Lightweight deployment health check (no DB/model overhead)."""
    return jsonify({"status": "ok"}), 200

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('landing'))

@app.route('/landing')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/ppt')
def ppt():
    ppt_dir = BASE_DIR / 'ppt_like_work'
    return send_from_directory(str(ppt_dir), 'index.html')

@app.route('/media/<path:filename>')
def serve_media(filename):
    media_dir = BASE_DIR / 'front_end' / 'images_videos'
    return send_from_directory(str(media_dir), filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(str(BASE_DIR / 'front_end' / 'front_end_png'), filename)


# ==========================================
# ROUTES: AUTHENTICATION
# ==========================================
@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_pass = request.form.get('confirm_password', '')

        if not email or not password:
            flash('Email and password are required.', 'danger')
            return redirect(url_for('register'))

        if '@' not in email:
            flash('Please enter a valid email address.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_pass:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'warning')
            return redirect(url_for('register'))

        user = User(
            name=full_name if full_name else 'Cat Lover',
            email=email
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please enter email and password.', 'danger')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password, bcrypt_instance=bcrypt):
            login_user(user)
            flash('Welcome back to CatSense AI!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ==========================================
# ROUTES: SCAN THUMBNAIL (OWNER ONLY)
# ==========================================
@app.route('/scan/<int:scan_id>/thumb')
@login_required
def scan_thumb(scan_id):
    """Serves 400px JPEG thumbnail directly from database with private cache header."""
    pred = db.session.get(Prediction, scan_id)
    if not pred:
        return "Scan not found", 404
    if pred.user_id != current_user.id:
        return "Forbidden: owner access only", 403
    if not pred.thumb_bytes:
        return "Thumbnail not available for this modality", 404

    response = make_response(send_file(
        io.BytesIO(pred.thumb_bytes),
        mimetype=pred.thumb_mime or 'image/jpeg'
    ))
    response.headers['Cache-Control'] = 'private, max-age=86400'
    return response


# ==========================================
# ROUTES: CORE APPLICATION & INFERENCE
# ==========================================
@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_user_stats(current_user.id)
    return render_template('dashboard.html', **stats)


@app.route('/api/analyze', methods=['POST'])
@login_required
def api_analyze():
    """
    JSON API endpoint for interactive New Analysis page.
    Accepts multipart/form-data for image, audio, or both.
    Returns: {emotion, emoji, confidence, probabilities, scan_id, insight_html, insight_level}
    """
    try:
        modality = request.form.get('modality', 'image').lower()
        if modality not in ('image', 'audio', 'both', 'combined'):
            modality = 'image'

        raw_img_bytes = None
        thumb_bytes = None
        img_name = None
        p_v = None
        v_label, v_conf = None, None

        raw_aud_bytes = None
        aud_name = None
        p_a = None
        a_label, a_conf = None, None

        # 1. Process Image if requested
        if modality in ('image', 'both', 'combined'):
            img_file = request.files.get('image_file') or request.files.get('file')
            if not img_file or img_file.filename == '':
                return jsonify({"error": "An image file is required for this analysis."}), 400
            raw_img_bytes, thumb_bytes, img_name = extract_and_validate_image(img_file)
            p_v, v_label, v_conf = predict_image(raw_img_bytes, return_probs=True)

        # 2. Process Audio if requested
        if modality in ('audio', 'both', 'combined'):
            aud_file = request.files.get('audio_file') or (request.files.get('file') if modality == 'audio' else None)
            if not aud_file or aud_file.filename == '':
                return jsonify({"error": "An audio file is required for this analysis."}), 400
            raw_aud_bytes, aud_name = extract_and_validate_audio(aud_file)
            p_a, a_label, a_conf = predict_audio(raw_aud_bytes, return_probs=True)

        # 3. Determine Fused or Single Output
        if modality in ('both', 'combined'):
            fused_label, fused_conf, p_fused = fuse(p_v, p_a)
            final_emotion = fused_label
            final_conf_pct = float(fused_conf) * 100.0
            probabilities = {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_fused)}
            modalities_agree = (v_label == a_label)
            saved_filename = f"{img_name} + {aud_name}"
            saved_modality = "both"
            saved_file_type = "combined"
        elif modality == 'audio':
            final_emotion = a_label
            final_conf_pct = float(a_conf) * 100.0
            probabilities = {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_a)}
            modalities_agree = True
            saved_filename = aud_name
            saved_modality = "audio"
            saved_file_type = "audio"
            thumb_bytes = None
        else: # image
            final_emotion = v_label
            final_conf_pct = float(v_conf) * 100.0
            probabilities = {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_v)}
            modalities_agree = True
            saved_filename = img_name
            saved_modality = "image"
            saved_file_type = "image"

        # 4. Compute Biological Insight
        insight = get_insight(
            emotion=final_emotion,
            confidence_pct=final_conf_pct,
            probabilities=probabilities,
            modalities_agree=modalities_agree
        )

        # 5. Persist Prediction to Database (Confidence normalized to 0-100 float)
        pred_entry = Prediction(
            user_id=current_user.id,
            file_type=saved_file_type,
            filename=saved_filename,
            prediction_result=final_emotion,
            confidence=round(final_conf_pct, 2),
            thumb_bytes=thumb_bytes,
            thumb_mime='image/jpeg' if thumb_bytes else None,
            modality=saved_modality,
            probabilities_json=json.dumps(probabilities),
            insight_level=insight["level"]
        )
        db.session.add(pred_entry)
        db.session.commit()

        # Render insight card partial for seamless client injection
        insight_html = render_template('partials/insight_card.html', insight=insight)

        return jsonify({
            "success": True,
            "scan_id": pred_entry.id,
            "emotion": final_emotion,
            "emoji": EMOJI_MAP.get(final_emotion, "🐾"),
            "confidence": round(final_conf_pct, 1),
            "probabilities": probabilities,
            "insight_level": insight["level"],
            "insight_html": insight_html,
            "modalities": {
                "image": {"emotion": v_label, "confidence": round(v_conf * 100.0, 1)} if v_label else None,
                "audio": {"emotion": a_label, "confidence": round(a_conf * 100.0, 1)} if a_label else None
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.exception(f"Error in /api/analyze: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/new_analysis', methods=['GET', 'POST'])
@login_required
def new_analysis():
    """Handles both traditional form submission and renders interactive analysis UI."""
    prediction_text = None
    confidence_score = None
    file_type_used = None
    image_pred = None
    audio_pred = None
    image_conf = None
    audio_conf = None
    insight = None

    if request.method == 'POST':
        file_type = request.form.get('file_type', 'image')
        try:
            if file_type == 'combined':
                img_file = request.files.get('image_file')
                aud_file = request.files.get('audio_file')
                if not img_file or not aud_file or img_file.filename == '' or aud_file.filename == '':
                    flash('Both image and audio files are required for combined analysis.', 'danger')
                    return redirect(request.url)

                raw_img, thumb_bytes, img_name = extract_and_validate_image(img_file)
                raw_aud, aud_name = extract_and_validate_audio(aud_file)

                p_v, v_label, v_conf = predict_image(raw_img, return_probs=True)
                p_a, a_label, a_conf = predict_audio(raw_aud, return_probs=True)
                fused_label, fused_conf, p_fused = fuse(p_v, p_a)

                prediction_text = fused_label
                confidence_score = float(fused_conf) * 100.0
                file_type_used = "combined"
                image_pred = v_label
                audio_pred = a_label
                image_conf = float(v_conf) * 100.0
                audio_conf = float(a_conf) * 100.0

                prob_dict = {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_fused)}
                insight = get_insight(
                    emotion=fused_label,
                    confidence_pct=confidence_score,
                    probabilities=p_fused,
                    modalities_agree=(v_label == a_label)
                )

                pred_entry = Prediction(
                    user_id=current_user.id,
                    file_type='combined',
                    filename=f"{img_name} + {aud_name}",
                    prediction_result=fused_label,
                    confidence=round(confidence_score, 2),
                    thumb_bytes=thumb_bytes,
                    thumb_mime='image/jpeg',
                    modality='both',
                    probabilities_json=json.dumps(prob_dict),
                    insight_level=insight["level"]
                )
                db.session.add(pred_entry)
                db.session.commit()
                flash('Multi-modal analysis complete!', 'success')

            else:
                target_file = request.files.get('file')
                if not target_file or target_file.filename == '':
                    flash('Please select a file to analyze.', 'danger')
                    return redirect(request.url)

                if file_type == 'image':
                    raw_img, thumb_bytes, img_name = extract_and_validate_image(target_file)
                    p_v, v_label, v_conf = predict_image(raw_img, return_probs=True)
                    prediction_text = v_label
                    confidence_score = float(v_conf) * 100.0
                    file_type_used = "Facial"
                    prob_dict = {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_v)}
                    insight = get_insight(v_label, confidence_score, probabilities=p_v, modalities_agree=True)

                    pred_entry = Prediction(
                        user_id=current_user.id,
                        file_type='image',
                        filename=img_name,
                        prediction_result=v_label,
                        confidence=round(confidence_score, 2),
                        thumb_bytes=thumb_bytes,
                        thumb_mime='image/jpeg',
                        modality='image',
                        probabilities_json=json.dumps(prob_dict),
                        insight_level=insight["level"]
                    )
                    db.session.add(pred_entry)
                    db.session.commit()
                    flash('Facial analysis complete!', 'success')

                elif file_type == 'audio':
                    raw_aud, aud_name = extract_and_validate_audio(target_file)
                    p_a, a_label, a_conf = predict_audio(raw_aud, return_probs=True)
                    prediction_text = a_label
                    confidence_score = float(a_conf) * 100.0
                    file_type_used = "Vocal"
                    prob_dict = {cls: round(float(p), 4) for cls, p in zip(CANONICAL_CLASSES, p_a)}
                    insight = get_insight(a_label, confidence_score, probabilities=p_a, modalities_agree=True)

                    pred_entry = Prediction(
                        user_id=current_user.id,
                        file_type='audio',
                        filename=aud_name,
                        prediction_result=a_label,
                        confidence=round(confidence_score, 2),
                        thumb_bytes=None,
                        thumb_mime=None,
                        modality='audio',
                        probabilities_json=json.dumps(prob_dict),
                        insight_level=insight["level"]
                    )
                    db.session.add(pred_entry)
                    db.session.commit()
                    flash('Vocal analysis complete!', 'success')

        except Exception as e:
            db.session.rollback()
            logging.exception(f"Error processing upload: {e}")
            flash(f"Error analyzing media: {e}", 'danger')

    return render_template(
        'new_analysis.html',
        prediction=prediction_text,
        confidence=confidence_score,
        file_type=file_type_used,
        image_prediction=image_pred,
        audio_prediction=audio_pred,
        image_confidence=image_conf,
        audio_confidence=audio_conf,
        insight=insight
    )


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Handles trial analyses for unauthenticated visitors."""
    if current_user.is_authenticated:
        return redirect(url_for('new_analysis'))

    prediction_text = None
    confidence_score = None
    file_type_used = None
    image_pred = None
    audio_pred = None
    image_conf = None
    audio_conf = None
    insight = None
    trial_used = session.get('trial_used', False)

    if request.method == 'POST':
        if trial_used:
            flash('Free trial limit reached. Please create an account to run unlimited analyses.', 'warning')
            return redirect(url_for('demo'))

        file_type = request.form.get('file_type', 'image')
        try:
            if file_type == 'combined':
                img_file = request.files.get('image_file')
                aud_file = request.files.get('audio_file')
                if not img_file or not aud_file or img_file.filename == '' or aud_file.filename == '':
                    flash('Both files are required for combined analysis.', 'danger')
                    return redirect(request.url)

                raw_img, _, _ = extract_and_validate_image(img_file)
                raw_aud, _ = extract_and_validate_audio(aud_file)

                p_v, v_label, v_conf = predict_image(raw_img, return_probs=True)
                p_a, a_label, a_conf = predict_audio(raw_aud, return_probs=True)
                fused_label, fused_conf, p_fused = fuse(p_v, p_a)

                prediction_text = fused_label
                confidence_score = float(fused_conf) * 100.0
                file_type_used = "combined"
                image_pred = v_label
                audio_pred = a_label
                image_conf = float(v_conf) * 100.0
                audio_conf = float(a_conf) * 100.0

                insight = get_insight(fused_label, confidence_score, probabilities=p_fused, modalities_agree=(v_label == a_label))
                session['trial_used'] = True
                trial_used = True
                flash('Demo multi-modal analysis complete!', 'success')

            else:
                target_file = request.files.get('file')
                if not target_file or target_file.filename == '':
                    flash('Please select a file.', 'danger')
                    return redirect(request.url)

                if file_type == 'image':
                    raw_img, _, _ = extract_and_validate_image(target_file)
                    p_v, v_label, v_conf = predict_image(raw_img, return_probs=True)
                    prediction_text = v_label
                    confidence_score = float(v_conf) * 100.0
                    file_type_used = "Facial"
                    insight = get_insight(v_label, confidence_score, probabilities=p_v, modalities_agree=True)
                else:
                    raw_aud, _ = extract_and_validate_audio(target_file)
                    p_a, a_label, a_conf = predict_audio(raw_aud, return_probs=True)
                    prediction_text = a_label
                    confidence_score = float(a_conf) * 100.0
                    file_type_used = "Vocal"
                    insight = get_insight(a_label, confidence_score, probabilities=p_a, modalities_agree=True)

                session['trial_used'] = True
                trial_used = True
                flash('Demo analysis complete!', 'success')

        except Exception as e:
            logging.exception(f"Demo analysis failed: {e}")
            flash(f"Error processing file: {e}", 'danger')

    return render_template(
        'demo.html',
        prediction=prediction_text,
        confidence=confidence_score,
        file_type=file_type_used,
        image_prediction=image_pred,
        audio_prediction=audio_pred,
        image_confidence=image_conf,
        audio_confidence=audio_conf,
        insight=insight,
        trial_used=trial_used
    )


# ==========================================
# ROUTES: METRICS & CHART.JS API
# ==========================================
@app.route('/metrics')
@login_required
def metrics():
    stats = get_user_stats(current_user.id)
    return render_template('metrics.html', **stats)


@app.route('/api/metrics')
@login_required
def api_metrics():
    """Returns owner-scoped analytics data formatted for Chart.js 4.4.x."""
    stats = get_user_stats(current_user.id)
    all_preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.asc()).all()

    # Timeline of scans (up to last 30 chronologically)
    timeline = []
    for p in all_preds[-30:]:
        timeline.append({
            "id": p.id,
            "date": p.timestamp.strftime("%b %d, %H:%M") if p.timestamp else "",
            "confidence": p.display_confidence or 0.0,
            "emotion": p.prediction_result.capitalize() if p.prediction_result else "Happy",
            "modality": p.modality or p.file_type or "image"
        })

    # Average confidence per emotion
    conf_by_emotion = {cls: [] for cls in CANONICAL_CLASSES}
    for p in all_preds:
        res = p.prediction_result.capitalize() if p.prediction_result else None
        if res in conf_by_emotion and p.display_confidence is not None:
            conf_by_emotion[res].append(p.display_confidence)

    avg_conf_per_emotion = {
        cls: round(sum(vals) / len(vals), 1) if vals else 0.0
        for cls, vals in conf_by_emotion.items()
    }

    return jsonify({
        "emotion_distribution": stats['emotion_distribution'],
        "average_confidence_per_emotion": avg_conf_per_emotion,
        "timeline": timeline,
        "modality_counts": {
            "image": stats['image_count'],
            "audio": stats['audio_count'],
            "both": stats['combined_count']
        },
        "total_scans": stats['total_predictions'],
        "overall_avg_confidence": stats['overall_avg_confidence']
    }), 200


# ==========================================
# ROUTES: HISTORY (FILTERS & PAGINATION)
# ==========================================
@app.route('/history')
@login_required
def history():
    """Displays user's prediction history with filters, search, and pagination."""
    filter_type = request.args.get('type', 'all').lower()
    filter_emotion = request.args.get('emotion', 'all').capitalize()
    search_query = request.args.get('q', '').strip()

    try:
        page = max(1, int(request.args.get('page', 1)))
    except ValueError:
        page = 1
    per_page = 10

    query = Prediction.query.filter_by(user_id=current_user.id)

    # Filter by modality
    if filter_type in ('image', 'facial'):
        query = query.filter((Prediction.modality == 'image') | (Prediction.file_type.in_(['image', 'Facial'])))
    elif filter_type in ('audio', 'vocal'):
        query = query.filter((Prediction.modality == 'audio') | (Prediction.file_type.in_(['audio', 'Vocal'])))
    elif filter_type in ('both', 'combined'):
        query = query.filter((Prediction.modality.in_(['both', 'combined'])) | (Prediction.file_type == 'combined'))

    # Filter by emotion
    if filter_emotion in CANONICAL_CLASSES:
        query = query.filter(Prediction.prediction_result.ilike(filter_emotion))

    # Search filter
    if search_query:
        query = query.filter(
            (Prediction.filename.ilike(f"%{search_query}%")) |
            (Prediction.prediction_result.ilike(f"%{search_query}%"))
        )

    total_count = query.count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages

    predictions = query.order_by(Prediction.timestamp.desc()) \
                       .offset((page - 1) * per_page) \
                       .limit(per_page) \
                       .all()

    return render_template(
        'history.html',
        predictions=predictions,
        current_page=page,
        total_pages=total_pages,
        total_count=total_count,
        selected_type=filter_type,
        selected_emotion=filter_emotion,
        search_query=search_query,
        canonical_classes=CANONICAL_CLASSES
    )


# ==========================================
# ROUTES: USER PROFILE
# ==========================================
@app.route('/profile')
@login_required
def profile():
    stats = get_user_stats(current_user.id)
    return render_template('profile.html', **stats)


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    name = request.form.get('name', '').strip()
    new_password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    if name:
        current_user.name = name

    if new_password:
        if len(new_password) < 6:
            flash('New password must be at least 6 characters long.', 'danger')
            return redirect(url_for('profile'))
        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('profile'))
        current_user.set_password(new_password)

    db.session.commit()
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))


# ==========================================
# SECURITY HEADERS & ERROR HANDLERS
# ==========================================
@app.after_request
def add_security_headers(response):
    """OWASP Security Headers."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500


# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("  MeowMind / CatSense AI - Production Server")
    print("  Starting at http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
