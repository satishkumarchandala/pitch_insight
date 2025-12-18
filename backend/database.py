"""
MongoDB Database Configuration and Models
Stores pitch analysis results for historical tracking
"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, List
from datetime import datetime
import os
from bson import ObjectId

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "pitch_insight")
ENABLE_DATABASE = os.getenv("ENABLE_DATABASE", "true").lower() == "true"

# Global database client
db_client: Optional[AsyncIOMotorClient] = None
database = None


class DatabaseManager:
    """Manages MongoDB connection and operations"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
    
    async def connect(self):
        """Connect to MongoDB Atlas"""
        if not ENABLE_DATABASE or not MONGODB_URI:
            print("⚠️  Database disabled or MONGODB_URI not set")
            return False
        
        try:
            self.client = AsyncIOMotorClient(MONGODB_URI)
            # Test connection
            await self.client.admin.command('ping')
            self.db = self.client[MONGODB_DB_NAME]
            self.connected = True
            print("✅ Connected to MongoDB Atlas")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.connected = False
            print("🔌 Disconnected from MongoDB")
    
    async def save_analysis(self, analysis_data: Dict) -> Optional[str]:
        """
        Save pitch analysis result to database
        
        Args:
            analysis_data: Complete analysis result dictionary
            
        Returns:
            Document ID as string, or None if save failed
        """
        if not self.connected:
            return None
        
        try:
            # Add metadata
            document = {
                **analysis_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Insert into analyses collection
            result = await self.db.analyses.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
            return None
    
    async def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """
        Retrieve analysis by ID
        
        Args:
            analysis_id: MongoDB ObjectId as string
            
        Returns:
            Analysis document or None if not found
        """
        if not self.connected:
            return None
        
        try:
            result = await self.db.analyses.find_one({"_id": ObjectId(analysis_id)})
            if result:
                result["_id"] = str(result["_id"])
            return result
        except Exception as e:
            print(f"❌ Failed to retrieve analysis: {e}")
            return None
    
    async def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """
        Get recent analyses
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of analysis documents
        """
        if not self.connected:
            return []
        
        try:
            cursor = self.db.analyses.find().sort("created_at", -1).limit(limit)
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for result in results:
                result["_id"] = str(result["_id"])
            
            return results
        except Exception as e:
            print(f"❌ Failed to retrieve analyses: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.connected:
            return {"total_analyses": 0, "connected": False}
        
        try:
            total = await self.db.analyses.count_documents({})
            
            # Get pitch type distribution
            pipeline = [
                {
                    "$group": {
                        "_id": "$final_classification.predicted_type",
                        "count": {"$sum": 1}
                    }
                }
            ]
            pitch_types = await self.db.analyses.aggregate(pipeline).to_list(length=100)
            
            return {
                "total_analyses": total,
                "pitch_type_distribution": {item["_id"]: item["count"] for item in pitch_types},
                "connected": True
            }
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {"error": str(e), "connected": False}


# Global database manager instance
db_manager = DatabaseManager()


async def get_database():
    """Dependency for FastAPI endpoints"""
    return db_manager
