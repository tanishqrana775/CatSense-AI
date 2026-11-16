"""
Test script to verify database functionality
Run this to check if predictions are being saved correctly
"""
import os
from pathlib import Path
from app import app, db, User, Prediction
from datetime import datetime, timezone

def test_database():
    """Test database operations"""
    with app.app_context():
        print("=" * 60)
        print("  DATABASE TEST - MeowMind")
        print("=" * 60)
        
        # Check if database file exists
        db_path = Path(__file__).parent / 'database' / 'cat_emotion.db'
        print(f"\n1. Database file exists: {db_path.exists()}")
        print(f"   Location: {db_path}")
        
        # Check tables
        print("\n2. Checking database tables...")
        try:
            user_count = User.query.count()
            pred_count = Prediction.query.count()
            print(f"   ✓ Users table: {user_count} users")
            print(f"   ✓ Predictions table: {pred_count} predictions")
        except Exception as e:
            print(f"   ✗ Error accessing tables: {e}")
            return
        
        # List all users
        print("\n3. Registered Users:")
        users = User.query.all()
        if users:
            for user in users:
                print(f"   - {user.email} (ID: {user.id}, Name: {user.name})")
        else:
            print("   No users found. Please register an account first.")
            return
        
        # List all predictions
        print("\n4. All Predictions:")
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
        if predictions:
            for pred in predictions:
                user = User.query.get(pred.user_id)
                print(f"   - {pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"User: {user.email if user else 'Unknown'} | "
                      f"Type: {pred.file_type} | "
                      f"File: {pred.filename} | "
                      f"Emotion: {pred.prediction_result}")
        else:
            print("   No predictions found. Try uploading a file in 'New Analysis'.")
        
        # Test creating a sample prediction
        print("\n5. Testing Prediction Creation...")
        if users:
            test_user = users[0]
            try:
                test_pred = Prediction(
                    user_id=test_user.id,
                    file_type='image',
                    filename='test_cat.jpg',
                    prediction_result='Happy',
                    timestamp=datetime.now(timezone.utc)
                )
                db.session.add(test_pred)
                db.session.commit()
                print(f"   ✓ Test prediction created successfully!")
                print(f"   ✓ Prediction ID: {test_pred.id}")
                
                # Verify it was saved
                saved_pred = Prediction.query.get(test_pred.id)
                if saved_pred:
                    print(f"   ✓ Verified: Prediction retrieved from database")
                    # Clean up test prediction
                    db.session.delete(saved_pred)
                    db.session.commit()
                    print(f"   ✓ Test prediction cleaned up")
                else:
                    print(f"   ✗ Could not retrieve test prediction")
                    
            except Exception as e:
                print(f"   ✗ Error creating test prediction: {e}")
                db.session.rollback()
        
        # Check predictions per user
        print("\n6. Predictions Per User:")
        for user in users:
            user_preds = Prediction.query.filter_by(user_id=user.id).all()
            print(f"   - {user.email}: {len(user_preds)} predictions")
            if user_preds:
                emotions = {}
                for pred in user_preds:
                    emotions[pred.prediction_result] = emotions.get(pred.prediction_result, 0) + 1
                print(f"     Emotions: {emotions}")
        
        print("\n" + "=" * 60)
        print("  DATABASE TEST COMPLETE")
        print("=" * 60)
        print("\nIf you see predictions above, the database is working!")
        print("If not, try:")
        print("  1. Register a new account")
        print("  2. Upload a cat image in 'New Analysis'")
        print("  3. Run this test again")
        print("\nTo reset the database:")
        print("  1. Stop the Flask app")
        print("  2. Delete database/cat_emotion.db")
        print("  3. Restart the app (tables will be recreated)")
        print("=" * 60)

if __name__ == '__main__':
    test_database()
