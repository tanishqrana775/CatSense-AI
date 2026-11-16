from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True, default='Cat Lover')
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    def set_password(self, password):
        """Hashes password using werkzeug.security.generate_password_hash."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password, bcrypt_instance=None):
        """
        Validates password against werkzeug hash.
        Falls back to bcrypt if present (and transparently upgrades to werkzeug hash).
        """
        try:
            if check_password_hash(self.password_hash, password):
                return True
        except Exception:
            pass

        if bcrypt_instance:
            try:
                if bcrypt_instance.check_password_hash(self.password_hash, password):
                    # Seamlessly upgrade legacy bcrypt hash to werkzeug hash
                    self.set_password(password)
                    db.session.commit()
                    return True
            except Exception:
                pass

        return False

    @property
    def display_name(self):
        """Return user's name, or derive from email if not set."""
        if self.name and self.name != 'Cat Lover':
            return self.name
        return self.email.split('@')[0].capitalize()

    def __repr__(self):
        return f"User('{self.name}', '{self.email}')"

class Prediction(db.Model):
    __tablename__ = 'prediction'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_type = db.Column(db.String(10), nullable=False) # 'image', 'audio', 'combined'
    filename = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True) # Normalized to 0-100 float
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Additive fields for production & multi-modal insight engine
    thumb_bytes = db.Column(db.LargeBinary, nullable=True)
    thumb_mime = db.Column(db.String(50), nullable=True)
    modality = db.Column(db.String(20), nullable=True) # 'image', 'audio', 'both'
    probabilities_json = db.Column(db.Text, nullable=True)
    insight_level = db.Column(db.String(30), nullable=True) # 'high', 'moderate', 'conflicted'

    @property
    def display_confidence(self):
        """Returns confidence guaranteed in 0-100 percentage range."""
        if self.confidence is None:
            return None
        # Handle legacy 0-1 float values
        if self.confidence <= 1.0 and self.confidence > 0:
            return round(self.confidence * 100.0, 1)
        return round(self.confidence, 1)

    def __repr__(self):
        return f"Prediction('{self.filename}', '{self.prediction_result}', conf={self.confidence})"

# Alias for compatibility with both Scan and Prediction terminology
Scan = Prediction

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        # Additive schema sync for existing SQLite databases
        try:
            from sqlalchemy import inspect, text
            inspector = inspect(db.engine)
            if 'prediction' in inspector.get_table_names():
                existing_cols = {col['name'] for col in inspector.get_columns('prediction')}
                add_cols = [
                    ('thumb_bytes', 'BLOB'),
                    ('thumb_mime', 'VARCHAR(50)'),
                    ('modality', 'VARCHAR(20)'),
                    ('probabilities_json', 'TEXT'),
                    ('insight_level', 'VARCHAR(30)')
                ]
                with db.engine.connect() as conn:
                    for col_name, col_type in add_cols:
                        if col_name not in existing_cols:
                            conn.execute(text(f"ALTER TABLE prediction ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
        except Exception as e:
            pass
