"""
One-off Idempotent Migration Script: SQLite -> PostgreSQL (or Target DATABASE_URL).
Copies users and predictions/scans from local cat_emotion.db to DATABASE_URL.
Normalizes all legacy 0.0-1.0 confidence values to 0.0-100.0 scale.
Idempotent: safe to run multiple times without duplicating records.
"""
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from database.database_logic import db, User, Prediction
from flask import Flask

def migrate(source_sqlite_path=None, target_db_url=None):
    if source_sqlite_path is None:
        source_sqlite_path = PROJECT_ROOT / "database" / "cat_emotion.db"
    
    if not Path(source_sqlite_path).exists():
        print(f"[ERROR] Source SQLite database not found at: {source_sqlite_path}")
        return False

    if target_db_url is None:
        target_db_url = os.environ.get("TARGET_DATABASE_URL") or os.environ.get("DATABASE_URL")
    
    if not target_db_url:
        print("[WARNING] No target DATABASE_URL specified. Migrating in-place on local SQLite database.")
        target_db_url = f"sqlite:///{source_sqlite_path}"
    elif target_db_url.startswith("postgres://"):
        target_db_url = target_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif target_db_url.startswith("postgresql://") and not target_db_url.startswith("postgresql+psycopg2://"):
        target_db_url = target_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    print("=" * 65)
    print("MEOWMIND / CATSENSE DATABASE MIGRATION")
    print(f"Source: {source_sqlite_path}")
    print(f"Target: {target_db_url.split('@')[-1] if '@' in target_db_url else target_db_url}")
    print("=" * 65)

    # 1. Inspect source database via direct sqlite3 connection
    src_conn = sqlite3.connect(str(source_sqlite_path))
    src_cursor = src_conn.cursor()

    src_users = src_cursor.execute("SELECT id, name, email, password_hash, created_at FROM user").fetchall()
    src_preds = src_cursor.execute("SELECT id, user_id, file_type, filename, prediction_result, confidence, timestamp FROM prediction").fetchall()
    print(f"[Source Stats] Found {len(src_users)} users and {len(src_preds)} predictions.")

    # 2. Setup Flask app with target DB URL
    temp_app = Flask(__name__)
    temp_app.config["SQLALCHEMY_DATABASE_URI"] = target_db_url
    temp_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(temp_app)

    with temp_app.app_context():
        # Ensure target schema exists
        db.create_all()

        target_user_count_before = User.query.count()
        target_pred_count_before = Prediction.query.count()
        print(f"[Target Before] Users: {target_user_count_before}, Predictions: {target_pred_count_before}")

        user_id_map = {} # source_id -> target_id

        # 3. Migrate Users (idempotent by email)
        for u_id, name, email, pwd_hash, created_at in src_users:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                user_id_map[u_id] = existing_user.id
            else:
                try:
                    c_time = datetime.fromisoformat(created_at) if isinstance(created_at, str) else datetime.utcnow()
                except Exception:
                    c_time = datetime.utcnow()

                new_user = User(
                    name=name,
                    email=email,
                    password_hash=pwd_hash,
                    created_at=c_time
                )
                db.session.add(new_user)
                db.session.flush() # Populate new_user.id
                user_id_map[u_id] = new_user.id

        db.session.commit()

        # 4. Migrate Predictions (idempotent, normalizes confidence to 0-100)
        migrated_preds_count = 0
        normalized_conf_count = 0

        for p_id, u_id, f_type, fname, result, conf, ts in src_preds:
            target_uid = user_id_map.get(u_id)
            if not target_uid:
                continue

            try:
                t_time = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.utcnow()
            except Exception:
                t_time = datetime.utcnow()

            # Normalize confidence from 0-1 to 0-100
            norm_conf = conf
            if conf is not None:
                if 0.0 < conf <= 1.0:
                    norm_conf = round(conf * 100.0, 2)
                    normalized_conf_count += 1
                else:
                    norm_conf = round(conf, 2)

            # Idempotence check: check if record already exists
            existing_pred = Prediction.query.filter_by(
                user_id=target_uid,
                filename=fname,
                timestamp=t_time
            ).first()

            if not existing_pred:
                new_pred = Prediction(
                    user_id=target_uid,
                    file_type=f_type,
                    filename=fname,
                    prediction_result=result,
                    confidence=norm_conf,
                    timestamp=t_time,
                    modality="both" if f_type == "combined" else f_type
                )
                db.session.add(new_pred)
                migrated_preds_count += 1

        db.session.commit()

        target_user_count_after = User.query.count()
        target_pred_count_after = Prediction.query.count()

        print("\n" + "=" * 65)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print(f"Target Users: {target_user_count_before} -> {target_user_count_after} (+{target_user_count_after - target_user_count_before} added)")
        print(f"Target Predictions: {target_pred_count_before} -> {target_pred_count_after} (+{migrated_preds_count} added)")
        print(f"Normalized Legacy Confidences (0-1 -> 0-100): {normalized_conf_count}")
        print("=" * 65)

    src_conn.close()
    return True

if __name__ == "__main__":
    migrate()
