"""Quick database check script"""
import sqlite3
from pathlib import Path

db_path = Path('database/cat_emotion.db')

if not db_path.exists():
    print("❌ Database file does not exist!")
    exit(1)

print("✅ Database file exists")
print(f"Location: {db_path.absolute()}\n")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check users
cursor.execute("SELECT COUNT(*) FROM user")
user_count = cursor.fetchone()[0]
print(f"👥 Users in database: {user_count}")

if user_count > 0:
    cursor.execute("SELECT id, email, name FROM user")
    users = cursor.fetchall()
    for user_id, email, name in users:
        print(f"   - ID: {user_id}, Email: {email}, Name: {name}")

# Check predictions
cursor.execute("SELECT COUNT(*) FROM prediction")
pred_count = cursor.fetchone()[0]
print(f"\n📊 Predictions in database: {pred_count}")

if pred_count > 0:
    cursor.execute("""
        SELECT p.id, p.user_id, p.file_type, p.filename, p.prediction_result, p.timestamp
        FROM prediction p
        ORDER BY p.timestamp DESC
        LIMIT 10
    """)
    predictions = cursor.fetchall()
    print("\nRecent predictions:")
    for pred_id, user_id, file_type, filename, result, timestamp in predictions:
        print(f"   - ID: {pred_id}, User: {user_id}, Type: {file_type}, File: {filename}, Result: {result}, Time: {timestamp}")
else:
    print("   No predictions found!")
    print("\n⚠️  This is why History and Metrics show 'no data'")
    print("\n📝 To fix:")
    print("   1. Make sure the Flask app is running")
    print("   2. Login to your account")
    print("   3. Go to 'New Analysis'")
    print("   4. Upload a cat image")
    print("   5. Check the Flask console for [DEBUG] messages")

conn.close()

print("\n" + "="*60)
