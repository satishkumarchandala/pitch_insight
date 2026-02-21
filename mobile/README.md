# Pitch Insight Mobile App

A beautiful, modern React Native mobile app for cricket pitch analysis powered by AI.

## Features

✨ **Modern UI/UX**
- Dark theme with beautiful gradients
- Smooth animations and transitions
- Intuitive navigation

🏏 **Pitch Analysis**
- Upload images from gallery
- Take photos with camera
- AI-powered pitch type detection
- Confidence scores and detailed probabilities

👤 **User Management**
- Secure authentication
- User profiles
- Analysis history
- Guest mode support

## Tech Stack

- **React Native** with Expo
- **React Navigation** for routing
- **Axios** for API calls
- **AsyncStorage** for local data
- **Expo Camera & Image Picker** for media
- **Linear Gradients** for beautiful UI

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Expo Go app on your mobile device

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Scan the QR code with:
   - **Android**: Expo Go app
   - **iOS**: Camera app (opens in Expo Go)

## Folder Structure

```
mobile/
├── src/
│   ├── components/     # Reusable UI components
│   ├── screens/        # App screens
│   ├── contexts/       # React contexts (Auth, etc.)
│   ├── services/       # API services
│   └── constants/      # Theme, colors, config
├── assets/            # Images, fonts, etc.
├── App.js            # Main app entry point
└── package.json      # Dependencies
```

## Screens

1. **Home**: Welcome screen with features overview
2. **Analysis**: Upload/capture images for pitch analysis
3. **Profile**: User account and settings
4. **Auth**: Login and registration

## API Integration

The app connects to the Pitch Insight backend at:
```
https://pitch-insight.onrender.com
```

Endpoints:
- `POST /api/auth/login` - User login
- `POST /api/auth/signup` - User registration
- `POST /api/analyze` - Pitch analysis

## Development

```bash
# Start development server
npm start

# Start with cache clear
npm start -- --clear

# Run on Android
npm run android

# Run on iOS  
npm run ios
```

## Troubleshooting

### Metro Bundler Issues
```bash
npx expo start --clear
```

### Port Already in Use
```bash
npx kill-port 8081
npm start
```

### Backend Connection Issues
- Make sure backend server is running
- Check API_BASE_URL in `src/constants/config.js`
- Backend has 60s timeout for cold starts

## Built With Excellence

- 🎨 Modern dark theme design
- ⚡ Fast and responsive
- 🔒 Secure authentication
- 📱 Cross-platform (iOS & Android)
- 🚀 Production-ready code

---

**Ready to analyze cricket pitches like never before!** 🏏
