"""
Script to clean up database and remove predictions with emotions 
that are not in the trained canonical set.
"""
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CANONICAL_CLASSES

TRAINED_EMOTIONS = CANONICAL_CLASSES

def clean_database():
    conn = sqlite3.connect('database/cat_emotion.db')
    cursor = conn.cursor()
    
    try:
        # Check current predictions
        cursor.execute("SELECT id, prediction_result FROM prediction")
        all_predictions = cursor.fetchall()
        
        print(f"Total predictions in database: {len(all_predictions)}")
        
        # Find predictions with invalid emotions
        invalid_predictions = [p for p in all_predictions if p[1] not in TRAINED_EMOTIONS]
        
        if invalid_predictions:
            print(f"\nFound {len(invalid_predictions)} predictions with invalid emotions:")
            emotion_counts = {}
            for pred in invalid_predictions:
                emotion = pred[1]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            for emotion, count in emotion_counts.items():
                print(f"  - {emotion}: {count} predictions")
            
            # Delete invalid predictions
            print("\nDeleting invalid predictions...")
            for pred in invalid_predictions:
                cursor.execute("DELETE FROM prediction WHERE id = ?", (pred[0],))
            
            conn.commit()
            print(f"✅ Deleted {len(invalid_predictions)} invalid predictions")
        else:
            print("✅ No invalid predictions found. Database is clean!")
        
        # Show remaining predictions
        cursor.execute("SELECT prediction_result, COUNT(*) FROM prediction GROUP BY prediction_result")
        remaining = cursor.fetchall()
        
        print(f"\nRemaining predictions by emotion:")
        for emotion, count in remaining:
            print(f"  - {emotion}: {count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting database cleanup...")
    print("=" * 60)
    clean_database()
    print("=" * 60)
    print("Database cleanup complete!")
