# User Analysis History Feature

## Overview
The application now includes a comprehensive user-specific analysis history system. Each user can view, retrieve, and delete their own pitch analyses without accessing other users' data.

## Database Storage

### Collections
- **users**: Stores user authentication data
- **analysis_history**: Stores all user analyses with complete data

### Analysis Record Structure
```javascript
{
  "user_id": "ObjectId as string",
  "user_email": "user@example.com",
  "analysis_id": "unique_analysis_id",
  "analysis_type": "complete" | "quick",
  "image_name": "pitch_image.jpg",
  "image_size": 2.5, // MB
  
  // Detection Results
  "pitch_detection": {
    "class": "pitch",
    "confidence": 0.95,
    "bbox": [x, y, width, height]
  },
  
  // Classification
  "pitch_type": "batting_friendly",
  "confidence": 0.87,
  
  // Features (Complete Analysis Only)
  "features": {
    "grass_coverage": "high",
    "cracks": "minimal",
    "moisture": "moderate",
    "hardness": "medium",
    // ... more features
  },
  
  // Classifications
  "ml_classification": {
    "prediction": "batting_friendly",
    "confidence": 0.85,
    "probabilities": { /* all class probabilities */ }
  },
  "final_classification": {
    "prediction": "batting_friendly",
    "confidence": 0.87
  },
  
  // Match Information
  "match_info": {
    "format": "test",
    "overs": 90,
    "format_description": "Test Match (90 overs per innings)"
  },
  
  // Weather Data (if included)
  "weather_included": true,
  "weather_data": {
    "location": {
      "name": "Mumbai",
      "region": "Maharashtra",
      "country": "India",
      "lat": 19.07,
      "lon": 72.87
    },
    "current": {
      "temperature": 28.5,
      "condition": "Partly cloudy",
      "humidity": 75,
      "wind_speed": 15.2,
      "wind_direction": "SW",
      "cloud_cover": 40,
      "uv_index": 6,
      "pressure": 1012
    },
    "forecast": {
      "sessions": [
        {
          "name": "Session 1",
          "avg_temp": 26.8,
          "max_humidity": 78,
          "avg_cloud": 45,
          "conditions": "Cloudy"
        }
        // ... more sessions
      ]
    },
    "impact_analysis": {
      "swing_score": 0.72,
      "spin_score": 0.58,
      "drying_rate": "moderate",
      "dew_likelihood": 0.65,
      "key_factors": ["High humidity favors swing", "..."]
    }
  },
  
  // Location
  "location": {
    "city": "Mumbai",
    "latitude": 19.07,
    "longitude": 72.87
  },
  
  // Match Strategy
  "match_strategy": {
    "optimal_approach": "batting_first",
    "key_considerations": ["Pitch likely to deteriorate", "..."],
    "toss_advantage": "batting"
  },
  
  // Metadata
  "processing_time": 5.23,
  "created_at": "2024-01-15T10:30:00Z",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

## Backend API Endpoints

### 1. Get User History List
**Endpoint:** `GET /api/auth/history`  
**Authentication:** Required (Bearer token)  
**Response:**
```json
{
  "history": [
    {
      "analysis_id": "ABC123",
      "image_name": "pitch1.jpg",
      "pitch_type": "batting_friendly",
      "confidence": 0.87,
      "match_info": {
        "format": "test",
        "overs": 90,
        "format_description": "Test Match (90 overs per innings)"
      },
      "weather_included": true,
      "location": {
        "city": "Mumbai",
        "latitude": 19.07,
        "longitude": 72.87
      },
      "processing_time": 5.23,
      "created_at": "2024-01-15T10:30:00Z",
      "timestamp": "2024-01-15T10:30:00.123456"
    }
    // ... more analyses (up to 100 most recent)
  ]
}
```
**Features:**
- Returns summary of last 100 analyses
- Sorted by newest first
- User-restricted (only own analyses)

### 2. Get Detailed Analysis
**Endpoint:** `GET /api/auth/history/{analysis_id}`  
**Authentication:** Required (Bearer token)  
**Response:**
```json
{
  // Complete analysis record with all fields
  "user_id": "...",
  "analysis_id": "ABC123",
  "pitch_detection": { /* ... */ },
  "features": { /* ... */ },
  "weather_data": { /* ... */ },
  // ... all fields
}
```
**Features:**
- Returns complete analysis data
- User ownership verified
- 404 if not found or unauthorized

### 3. Delete Analysis
**Endpoint:** `DELETE /api/auth/history/{analysis_id}`  
**Authentication:** Required (Bearer token)  
**Response:**
```json
{
  "message": "Analysis deleted successfully",
  "analysis_id": "ABC123"
}
```
**Features:**
- User ownership verified before deletion
- 404 if not found or unauthorized

### 4. Analyze Endpoints (Modified)
Both `/api/analyze` and `/api/quick-analyze` now save to history if user is authenticated:

**Quick Analysis:**
- Analysis type: "quick"
- Saves basic classification results
- No weather or detailed features

**Complete Analysis:**
- Analysis type: "complete"
- Saves full analysis with all data
- Includes weather, features, strategy

## Frontend Components

### HistorySection Component
**Location:** `frontend/src/components/HistorySection.jsx`

**Features:**
- Grid layout of analysis cards
- Summary view with key information
- "View Details" button for full analysis
- "Delete" button with confirmation
- Loading and error states
- Empty state for no history

**Props:**
```javascript
{
  authToken: string,        // JWT token for API calls
  onViewDetails: function   // Callback when viewing details
}
```

### Integration in Analysis Page
**Location:** `frontend/src/pages/Analysis.jsx`

**Tab Structure:**
1. **New Analysis Tab**
   - Quick/Complete analysis options
   - Upload and configuration
   - Results display

2. **History Tab**
   - List of past analyses
   - View details action
   - Delete action

## User Flow

### Performing Analysis
1. User logs in
2. Goes to Analysis page
3. Chooses Quick or Complete analysis
4. Uploads image and configures options
5. Analysis is performed
6. Results are displayed
7. **Analysis is automatically saved to database**

### Viewing History
1. User goes to Analysis page
2. Clicks "History" tab
3. Sees grid of past analyses
4. Each card shows:
   - Image name
   - Pitch type with color badge
   - Confidence percentage
   - Match format
   - Location (if available)
   - Weather indicator
   - Timestamp
   - Processing time

### Viewing Details
1. Click "View Details" (eye icon) on any history card
2. Full analysis is fetched from database
3. Results page is displayed with complete data
4. Can navigate back to history

### Deleting Analysis
1. Click "Delete" (trash icon) on history card
2. Confirmation dialog appears
3. If confirmed, analysis is deleted
4. Card is removed from view

## Security & Privacy

### User Isolation
- All queries filter by `user_id`
- Users cannot access other users' analyses
- MongoDB queries enforce: `{"user_id": str(current_user["_id"])}`

### Authentication
- JWT tokens required for all history endpoints
- Token verified on every request
- User identity extracted from token

### Ownership Verification
- Detail retrieval checks user_id match
- Deletion checks user_id match
- 404 returned if not found or unauthorized

## Styling

### Color Coding
- **Batting Friendly:** Green (#4CAF50)
- **Bowling Friendly:** Red (#FF5722)
- **Spin Friendly:** Orange (#FF9800)
- **Seam Friendly:** Blue (#2196F3)

### Responsive Design
- Desktop: Grid with multiple columns
- Mobile: Single column layout
- Touch-friendly buttons
- Optimized for all screen sizes

## Data Retention
- No automatic deletion
- User can manually delete analyses
- All analyses stored indefinitely unless deleted
- Future enhancement: Add data retention policies

## Future Enhancements
1. **Export Analysis**
   - Download as PDF
   - Export to CSV
   - Share via link

2. **Filtering & Search**
   - Filter by pitch type
   - Filter by date range
   - Search by location
   - Filter by match format

3. **Statistics Dashboard**
   - Analysis count by pitch type
   - Average confidence scores
   - Location-based insights
   - Time-based trends

4. **Comparison Feature**
   - Compare multiple analyses
   - Side-by-side view
   - Difference highlighting

5. **Notes & Tags**
   - Add custom notes to analyses
   - Tag analyses for organization
   - Search by tags

## Testing

### Manual Testing Checklist
- [ ] Create analysis as authenticated user
- [ ] Verify analysis appears in history
- [ ] View analysis details
- [ ] Delete analysis
- [ ] Verify deleted analysis is removed
- [ ] Test with multiple users (isolation)
- [ ] Test pagination (more than 100 analyses)
- [ ] Test with quick vs complete analyses
- [ ] Test without authentication
- [ ] Test with invalid analysis_id

### API Testing
Use the provided test script in `backend/test_api.py` to test all endpoints.

## Troubleshooting

### History Not Loading
- Check if user is authenticated (token exists)
- Check MongoDB connection
- Check browser console for errors
- Verify API endpoint is accessible

### Analysis Not Appearing in History
- Verify user was authenticated during analysis
- Check if analysis completed successfully
- Check MongoDB collection for record
- Verify user_id matches

### Cannot Delete Analysis
- Verify user owns the analysis
- Check authentication token
- Check MongoDB connection
- Verify analysis_id is correct

## Technical Notes

### MongoDB Indexes
Recommended indexes for performance:
```javascript
db.analysis_history.createIndex({ "user_id": 1, "created_at": -1 })
db.analysis_history.createIndex({ "analysis_id": 1, "user_id": 1 })
```

### Storage Considerations
- Average analysis size: ~5-10 KB (without image)
- 1000 analyses ≈ 5-10 MB
- Images are not stored (only metadata)
- Weather data adds ~2-3 KB per analysis

### Performance
- History list query: <100ms (with indexes)
- Detail retrieval: <50ms
- Deletion: <50ms
- Analysis save: <100ms (async)
