"""
Migration script to add created_at column to User table
Run this once to update your existing database
"""
import sqlite3
from datetime import datetime, timezone

def migrate_database():
    conn = sqlite3.connect('database/cat_emotion.db')
    cursor = conn.cursor()
    
    try:
        # Check if created_at column already exists
        cursor.execute("PRAGMA table_info(user)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'created_at' not in columns:
            print("Adding created_at column to user table...")
            
            # Add the created_at column with a default value
            default_date = datetime.now(timezone.utc).isoformat()
            cursor.execute(f"""
                ALTER TABLE user 
                ADD COLUMN created_at TIMESTAMP DEFAULT '{default_date}'
            """)
            
            # Update existing users with current timestamp
            cursor.execute(f"""
                UPDATE user 
                SET created_at = '{default_date}' 
                WHERE created_at IS NULL
            """)
            
            conn.commit()
            print("✅ Migration completed successfully!")
            print(f"   All existing users now have created_at set to: {default_date}")
        else:
            print("✅ created_at column already exists. No migration needed.")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("Starting database migration...")
    migrate_database()
