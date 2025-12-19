"""
MongoDB Database Manager
Handles database connections and operations for Pitch Insight
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MongoDB connections and operations"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.connected = False
        self.enabled = os.getenv("ENABLE_DATABASE", "false").lower() == "true"
        
    async def connect(self):
        """Connect to MongoDB"""
        if not self.enabled:
            logger.info("Database disabled via ENABLE_DATABASE environment variable")
            return
            
        try:
            mongodb_uri = os.getenv("MONGODB_URI")
            mongodb_db_name = os.getenv("MONGODB_DB_NAME", "pitch_insight")
            
            if not mongodb_uri:
                logger.warning("MONGODB_URI not configured, database features disabled")
                return
                
            self.client = AsyncIOMotorClient(mongodb_uri)
            self.db = self.client[mongodb_db_name]
            
            # Test connection
            await self.client.admin.command('ping')
            self.connected = True
            logger.info(f"Connected to MongoDB database: {mongodb_db_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
            
    async def save_analysis(self, analysis_data: Dict) -> Optional[str]:
        """
        Save pitch analysis to database
        
        Args:
            analysis_data: Complete analysis result dictionary
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.connected:
            return None
            
        try:
            # Add timestamp
            doc = {
                **analysis_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db.analyses.insert_one(doc)
            logger.info(f"Analysis saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return None
            
    async def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """
        Retrieve analysis by ID
        
        Args:
            analysis_id: Document ID
            
        Returns:
            Analysis document or None
        """
        if not self.connected:
            return None
            
        try:
            from bson import ObjectId
            result = await self.db.analyses.find_one({"_id": ObjectId(analysis_id)})
            
            if result:
                result["_id"] = str(result["_id"])
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis: {e}")
            return None
            
    async def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """
        Get recent analyses
        
        Args:
            limit: Number of results to return
            
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
            logger.error(f"Failed to retrieve recent analyses: {e}")
            return []
            
    async def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.connected:
            return {
                "connected": False,
                "total_analyses": 0,
                "pitch_type_distribution": {}
            }
            
        try:
            total = await self.db.analyses.count_documents({})
            
            # Get pitch type distribution
            pipeline = [
                {"$group": {
                    "_id": "$pitch_classification.predicted_class",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            distribution = {}
            async for doc in self.db.analyses.aggregate(pipeline):
                if doc["_id"]:
                    distribution[doc["_id"]] = doc["count"]
            
            return {
                "connected": True,
                "total_analyses": total,
                "pitch_type_distribution": distribution
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}")
            return {
                "connected": False,
                "total_analyses": 0,
                "pitch_type_distribution": {},
                "error": str(e)
            }


# Global database manager instance
db_manager = DatabaseManager()
