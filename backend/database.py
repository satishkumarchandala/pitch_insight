"""
MongoDB Database Configuration
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional
from config import MONGODB_URL, DATABASE_NAME

# Global database connection
_client: Optional[MongoClient] = None
_db = None


def get_database():
    """
    Get MongoDB database instance
    Creates connection on first call (lazy initialization)
    """
    global _client, _db
    
    if _db is None:
        try:
            _client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
            # Test connection
            _client.admin.command('ping')
            _db = _client[DATABASE_NAME]
            print(f"✓ Connected to MongoDB: {DATABASE_NAME}")
            
            # Create indexes for users collection
            _db.users.create_index("email", unique=True)
            _db.users.create_index("username", unique=True)
            
        except ConnectionFailure as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            print("Make sure MongoDB is running on your system")
            raise
    
    return _db


def close_database_connection():
    """Close MongoDB connection"""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        print("✓ MongoDB connection closed")


# Collections
def get_users_collection():
    """Get users collection"""
    db = get_database()
    return db.users


def get_analysis_collection():
    """Get analysis history collection"""
    db = get_database()
    return db.analysis_history
