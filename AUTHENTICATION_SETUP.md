# Authentication Setup Guide

## Overview

This guide covers the newly added authentication system with MongoDB integration for the Pitch Insight application.

## Features Added

✅ User registration (signup)
✅ User login with JWT tokens
✅ Password hashing with bcrypt
✅ MongoDB for user storage
✅ Optional authentication (users can use the app without logging in)
✅ Analysis history tracking for authenticated users
✅ Protected routes
✅ Persistent sessions (token stored in localStorage)

## Prerequisites

### 1. Install MongoDB

**Windows:**
- Download MongoDB Community Server from: https://www.mongodb.com/try/download/community
- Install and start MongoDB service
- Default connection: `mongodb://localhost:27017/`

**Alternative: MongoDB Atlas (Cloud)**
- Sign up at: https://www.mongodb.com/cloud/atlas
- Create a free cluster
- Get connection string and update backend configuration

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

New packages installed:
- `pymongo>=4.6.0` - MongoDB driver
- `bcrypt>=4.1.0` - Password hashing
- `python-jose[cryptography]>=3.3.0` - JWT token handling
- `passlib>=1.7.4` - Password utilities
- `python-multipart>=0.0.6` - Form data handling

## Backend Configuration

### Environment Variables (Optional)

Create a `.env` file in the backend directory:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017/
DATABASE_NAME=pitch_insight

# Security (IMPORTANT: Change in production!)
SECRET_KEY=your-secret-key-change-this-in-production-123456789

# Weather API (if you have one)
WEATHER_API_KEY=your_openweathermap_api_key
```

### Default Configuration

If no `.env` file is provided, the system uses these defaults:
- MongoDB URL: `mongodb://localhost:27017/`
- Database: `pitch_insight`
- Secret Key: Default (CHANGE IN PRODUCTION)

## Database Structure

### Collections

1. **users** - User accounts
   - `_id`: ObjectId
   - `username`: string (unique)
   - `email`: string (unique)
   - `full_name`: string (optional)
   - `hashed_password`: string
   - `created_at`: datetime
   - `is_active`: boolean

2. **analysis_history** - Analysis records for authenticated users
   - `user_id`: string
   - `analysis_id`: string
   - `image_name`: string
   - `pitch_type`: string
   - `confidence`: float
   - `weather_data`: object (optional)
   - `created_at`: datetime

## API Endpoints

### Authentication Endpoints

#### 1. Sign Up
```http
POST /api/auth/signup
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepass123",
  "full_name": "John Doe" // optional
}
```

**Response:**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "username": "johndoe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "created_at": "2025-12-27T10:30:00",
  "is_active": true
}
```

#### 2. Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "securepass123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 3. Get Current User
```http
GET /api/auth/me
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "username": "johndoe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "created_at": "2025-12-27T10:30:00",
  "is_active": true
}
```

#### 4. Get Analysis History
```http
GET /api/auth/history
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "count": 5,
  "history": [
    {
      "_id": "...",
      "user_id": "507f1f77bcf86cd799439011",
      "analysis_id": "PITCH_20251227_103045",
      "image_name": "pitch.jpg",
      "pitch_type": "Green Top",
      "confidence": 0.95,
      "weather_data": {...},
      "created_at": "2025-12-27T10:30:45"
    }
  ]
}
```

### Updated Analysis Endpoint

The `/api/analyze` endpoint now optionally accepts authentication:

```http
POST /api/analyze
Authorization: Bearer <token>  // Optional
Content-Type: multipart/form-data

image: <file>
latitude: <float>
longitude: <float>
city: <string>
include_weather: <boolean>
```

- If authenticated, the analysis is saved to user's history
- If not authenticated, analysis works but history is not saved

## Frontend Changes

### New Components

1. **Auth.jsx** - Login/Signup modal component
2. **Auth.css** - Authentication styling

### Updated Components

1. **Header.jsx** - Now shows login/logout buttons and user info
2. **App.jsx** - Manages authentication state
3. **UploadSection.jsx** - Passes auth token with requests

### User Flow

1. User visits the app (can use without login)
2. Clicks "Login" button in header
3. Modal appears with login/signup form
4. After login:
   - Token stored in localStorage
   - User info displayed in header
   - Analysis history is saved
5. User can logout anytime

## Running the Application

### 1. Start MongoDB

**Windows:**
```bash
# Check if MongoDB is running
mongosh
```

If not running:
```bash
# Start MongoDB as a service (Windows)
net start MongoDB
```

### 2. Start Backend

```bash
cd backend
python app.py
```

Backend will:
- Connect to MongoDB
- Create indexes for users collection
- Start on http://localhost:8000

### 3. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will start on http://localhost:5173

## Testing the Authentication

### 1. Sign Up a New User

```bash
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "test123456",
    "full_name": "Test User"
  }'
```

### 2. Login

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123456"
  }'
```

Save the `access_token` from the response.

### 3. Access Protected Route

```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <your_access_token>"
```

### 4. Test Analysis with Auth

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Authorization: Bearer <your_access_token>" \
  -F "image=@path/to/pitch.jpg" \
  -F "include_weather=false"
```

## Security Notes

⚠️ **IMPORTANT FOR PRODUCTION:**

1. **Change the SECRET_KEY**: Update `SECRET_KEY` in backend/auth.py or use environment variable
2. **Use HTTPS**: Deploy with SSL/TLS certificates
3. **Update CORS**: Change `allow_origins=["*"]` to specific frontend URL
4. **MongoDB Security**: 
   - Enable authentication
   - Use strong passwords
   - Restrict network access
5. **Token Expiry**: Default is 7 days, adjust in `auth.py`
6. **Password Policy**: Enforce stronger passwords in production

## Troubleshooting

### MongoDB Connection Issues

```
Failed to connect to MongoDB
```

**Solutions:**
- Check if MongoDB is running: `mongosh`
- Verify connection string in configuration
- Check firewall settings

### JWT Token Errors

```
Invalid authentication credentials
```

**Solutions:**
- Check if token is being sent in headers
- Verify token hasn't expired
- Ensure SECRET_KEY matches between requests

### User Already Exists

```
Email already registered
```

**Solutions:**
- Use a different email
- Reset the database if testing: `db.users.drop()`

## Database Management

### View Users in MongoDB

```bash
mongosh
use pitch_insight
db.users.find().pretty()
```

### View Analysis History

```bash
db.analysis_history.find().pretty()
```

### Clear Database (Development Only)

```bash
db.users.deleteMany({})
db.analysis_history.deleteMany({})
```

## Next Steps

Optional enhancements you could add:

1. **Email Verification** - Verify email addresses
2. **Password Reset** - Forgot password functionality
3. **Profile Management** - Update user profile
4. **Social Login** - Google/GitHub OAuth
5. **Rate Limiting** - Prevent abuse
6. **Refresh Tokens** - Long-lived sessions
7. **Admin Panel** - User management
8. **Analytics Dashboard** - View all analyses

## Support

For issues or questions:
- Check MongoDB logs
- Check backend console for errors
- Verify all dependencies are installed
- Ensure MongoDB is running

---

**Created:** December 27, 2025
**Version:** 1.0.0
