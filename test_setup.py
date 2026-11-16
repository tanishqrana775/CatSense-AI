#!/usr/bin/env python3
"""
CatSense AI Production Setup Verification Script
Checks runtime environment, dependencies, model weights, and database connectivity.
"""

import sys
from pathlib import Path

print("=" * 60)
print("  🐾 CatSense AI Production Setup Verification")
print("=" * 60)
print()

# Check Python version
print("✓ Checking Python version...")
if sys.version_info < (3, 10):
    print("  ❌ Python 3.10+ required. You have:", sys.version)
    sys.exit(1)
else:
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Check required production packages
print("\n✓ Checking required packages...")
required_packages = [
    'flask',
    'flask_sqlalchemy',
    'flask_login',
    'flask_bcrypt',
    'flask_wtf',
    'flask_limiter',
    'tensorflow',
    'PIL',
    'numpy',
    'librosa',
    'soundfile'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Check project structure
print("\n✓ Checking project structure...")
BASE_DIR = Path(__file__).resolve().parent

required_dirs = [
    'backend',
    'backend/inference',
    'database',
    'models',
    'front_end',
    'front_end/templates',
    'front_end/templates/partials',
    'front_end/static',
    'front_end/static/styles',
    'front_end/static/scripts'
]

required_files = [
    'app.py',
    'config.py',
    'database/database_logic.py',
    'backend/config.py',
    'backend/insights.py',
    'backend/inference/predictor.py',
    'backend/inference/model_loader.py',
    'front_end/templates/base.html',
    'front_end/templates/index.html',
    'front_end/templates/login.html',
    'front_end/templates/register.html',
    'front_end/templates/dashboard.html',
    'front_end/templates/new_analysis.html',
    'front_end/templates/demo.html',
    'front_end/templates/history.html',
    'front_end/templates/metrics.html',
    'front_end/templates/profile.html',
    'front_end/templates/partials/navbar.html',
    'front_end/templates/partials/insight_card.html',
    'front_end/static/styles/tokens.css',
    'front_end/static/styles/components.css',
    'front_end/static/styles/main.css',
    'front_end/static/scripts/theme.js'
]

all_exist = True
for dir_path in required_dirs:
    full_path = BASE_DIR / dir_path
    if full_path.exists():
        print(f"  ✅ {dir_path}/")
    else:
        print(f"  ❌ {dir_path}/ - MISSING")
        all_exist = False

for file_path in required_files:
    full_path = BASE_DIR / file_path
    if full_path.exists():
        print(f"  ✅ {file_path}")
    else:
        print(f"  ❌ {file_path} - MISSING")
        all_exist = False

if not all_exist:
    print("\n❌ Critical files or directories are missing!")
    sys.exit(1)

# Check AI model artifacts
print("\n✓ Checking AI model artifacts...")
model_files = [
    'models/vision_model.keras',
    'models/audio_model.keras',
    'models/calibration.json'
]

models_exist = True
for model_path in model_files:
    full_path = BASE_DIR / model_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {model_path} ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ {model_path} - NOT FOUND")
        models_exist = False

if not models_exist:
    print("\n❌ Trained model artifacts are missing!")
    sys.exit(1)

# Test database initialization
print("\n✓ Testing database initialization...")
try:
    from app import app, db
    with app.app_context():
        db.create_all()
    print("  ✅ Database initialized and connected successfully")
except Exception as e:
    print(f"  ❌ Database initialization failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  ✨ CatSense AI environment verified successfully!")
print("=" * 60)
