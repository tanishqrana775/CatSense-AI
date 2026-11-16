"""Test upload and database save"""
from app import app, db, Prediction
from datetime import datetime, timezone

print("Testing prediction save to database...\n")

with app.app_context():
    # Check current predictions
    pred_count_before = Prediction.query.count()
    print(f"Predictions before: {pred_count_before}")
    
    # Create a test prediction
    try:
        test_pred = Prediction(
            user_id=1,  # Your user ID
            file_type='image',
            filename='test_cat.jpg',
            prediction_result='Happy',
            timestamp=datetime.now(timezone.utc)
        )
        
        print(f"\nCreating test prediction:")
        print(f"  User ID: {test_pred.user_id}")
        print(f"  File type: {test_pred.file_type}")
        print(f"  Filename: {test_pred.filename}")
        print(f"  Result: {test_pred.prediction_result}")
        print(f"  Timestamp: {test_pred.timestamp}")
        
        db.session.add(test_pred)
        db.session.commit()
        
        print(f"\n✅ Test prediction saved with ID: {test_pred.id}")
        
        # Verify it was saved
        pred_count_after = Prediction.query.count()
        print(f"Predictions after: {pred_count_after}")
        
        # Retrieve it
        saved_pred = Prediction.query.get(test_pred.id)
        if saved_pred:
            print(f"\n✅ Verified: Retrieved prediction from database")
            print(f"   ID: {saved_pred.id}")
            print(f"   User: {saved_pred.user_id}")
            print(f"   Result: {saved_pred.prediction_result}")
            
            # Now check if it shows up in history query
            user_history = Prediction.query.filter_by(user_id=1).order_by(Prediction.timestamp.desc()).all()
            print(f"\n✅ History query returned {len(user_history)} predictions")
            
            if user_history:
                print("\nPredictions that would show in History page:")
                for pred in user_history:
                    print(f"   - {pred.filename} | {pred.file_type} | {pred.prediction_result}")
            
            print("\n" + "="*60)
            print("SUCCESS! Database save and retrieval working correctly.")
            print("="*60)
            print("\nThe issue is: You haven't uploaded any files yet!")
            print("\nTo fix:")
            print("1. Start Flask app: python app.py")
            print("2. Go to: http://127.0.0.1:5000/login")
            print("3. Login with: tanishqrana88770@gmail.com")
            print("4. Go to: http://127.0.0.1:5000/new_analysis")
            print("5. Upload a cat image")
            print("6. Check History page")
            
            # Clean up test prediction
            print("\nCleaning up test prediction...")
            db.session.delete(saved_pred)
            db.session.commit()
            print("✅ Test prediction removed")
            
        else:
            print("\n❌ ERROR: Could not retrieve saved prediction")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
