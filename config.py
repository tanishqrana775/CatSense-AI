"""
Production Configuration for MeowMind / CatSense AI.
Supports Render, Neon/Supabase PostgreSQL, and local SQLite development.
"""
import os
import sys
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Set TensorFlow / OpenMP environment variables before TF initialization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger("catsense.config")

class Config:
    FLASK_ENV = os.environ.get("FLASK_ENV", "development").lower()
    IS_PRODUCTION = FLASK_ENV == "production" or "RENDER" in os.environ

    # Secret Key
    _env_secret = os.environ.get("SECRET_KEY")
    if IS_PRODUCTION and not _env_secret:
        raise ValueError("CRITICAL: SECRET_KEY environment variable is required in production mode.")
    SECRET_KEY = _env_secret or "dev-secret-key-catsense-ai-change-in-prod"

    # Database URL
    raw_db_url = os.environ.get("DATABASE_URL")
    if raw_db_url:
        # Standardize PostgreSQL dialect for SQLAlchemy + psycopg2
        if raw_db_url.startswith("postgres://"):
            db_uri = raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif raw_db_url.startswith("postgresql://") and not raw_db_url.startswith("postgresql+psycopg2://"):
            db_uri = raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        else:
            db_uri = raw_db_url
        
        # Non-SQLite pooling options
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": 280,
            "pool_size": 3,
            "max_overflow": 2
        }
        logger.info("Using PostgreSQL Database Configuration.")
    else:
        # Development SQLite fallback
        sqlite_path = BASE_DIR / "database" / "cat_emotion.db"
        sqlite_path.parent.mkdir(exist_ok=True)
        db_uri = f"sqlite:///{sqlite_path}"
        SQLALCHEMY_ENGINE_OPTIONS = {}
        logger.warning(f"DATABASE_URL not set. Falling back to local SQLite: {sqlite_path}")

    SQLALCHEMY_DATABASE_URI = db_uri
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Maximum file upload size: 10 MB
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    # Security Cookies
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = IS_PRODUCTION

    # CSRF Protection
    WTF_CSRF_ENABLED = True
    WTF_CSRF_CHECK_DEFAULT = True

    # Static / Template paths
    TEMPLATE_FOLDER = str(BASE_DIR / "front_end" / "templates")
    STATIC_FOLDER = str(BASE_DIR / "front_end" / "static")
