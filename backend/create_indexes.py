"""
MongoDB Database Index Creation Script
Run this to add missing performance indexes
"""

from pymongo import ASCENDING, DESCENDING
from database import get_database

def create_indexes():
    """Create all necessary indexes for optimal performance"""
    db = get_database()
    
    print("Creating MongoDB indexes...")
    
    # Users collection indexes
    users = db["users"]
    
    # Existing unique indexes (ensure they exist)
    users.create_index([("email", ASCENDING)], unique=True, name="email_unique")
    users.create_index([("username", ASCENDING)], unique=True, name="username_unique")
    
    # Subscription expiration index for background cleanup jobs
    users.create_index([("subscription_end_date", ASCENDING)], name="subscription_expiration")
    
    print("✓ Users collection indexes created")
    
    # Analyses collection indexes
    analyses = db["analyses"]
    
    # Existing indexes
    analyses.create_index([("analysis_id", ASCENDING)], unique=True, name="analysis_id_unique")
    analyses.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)], name="user_created_at")
    analyses.create_index([("image_hash", ASCENDING)], name="image_hash")
    
    # NEW: Index for history queries (Performance Fix #9)
    analyses.create_index(
        [("user_id", ASCENDING), ("saved_at", DESCENDING)], 
        name="user_saved_analyses"
    )
    
    # NEW: Compound index for Pro history filtering
    analyses.create_index(
        [("user_id", ASCENDING), ("full_result", ASCENDING), ("saved_at", DESCENDING)],
        name="user_full_results"
    )
    
    # NEW: TTL index for auto-deletion after 90 days (GDPR compliance)
    analyses.create_index(
        [("created_at", ASCENDING)], 
        expireAfterSeconds=7776000,  # 90 days in seconds
        name="auto_expire_old_analyses"
    )
    
    print("✓ Analyses collection indexes created")
    
    # List all indexes
    print("\n📊 Current indexes:")
    print("\nUsers collection:")
    for index in users.list_indexes():
        print(f"  - {index['name']}: {index.get('key', {})}")
    
    print("\nAnalyses collection:")
    for index in analyses.list_indexes():
        print(f"  - {index['name']}: {index.get('key', {})}")
    
    print("\n✅ All indexes created successfully!")

if __name__ == "__main__":
    create_indexes()
