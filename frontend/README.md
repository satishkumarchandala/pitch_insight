# 🏏 Pitch Insight Frontend

Beautiful, responsive React frontend for cricket pitch analysis.

## ✨ Features

- **Modern UI Design** - Dark theme with gradient accents
- **Fully Responsive** - Works perfectly on all devices
- **Drag & Drop Upload** - Easy image upload with preview
- **Weather Integration** - Optional weather-based analysis
- **Real-time Results** - Beautiful visualization of analysis
- **Smooth Animations** - Professional transitions and effects

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

Frontend will open at: **http://localhost:3000**

## 📋 Prerequisites

- Node.js 16+ installed
- Backend API running at http://localhost:8000

## 🛠️ Build for Production

```bash
npm run build
```

Production files will be in `dist/` folder.

## 📱 Features Overview

### Upload Section
- Drag & drop image upload
- Image preview
- File size display
- Weather options (optional)
- Location input (city, lat/long)
- Quick vs Complete analysis

### Results Section
- **Main Prediction Card**
  - Pitch type with confidence
  - Visual confidence bar
  - Feature-based adjustments

- **Probabilities Chart**
  - All 4 pitch types
  - Color-coded bars

- **Pitch Features** (if available)
  - Grass coverage (%)
  - Crack analysis
  - Moisture level
  - Color profile
  - Texture type
  - Brightness level

- **Weather Conditions** (if included)
  - Temperature, humidity
  - Wind speed, rainfall
  - Weather impact analysis
  - Severity indicators

- **Match Strategy**
  - Toss decision
  - Batting strategy (3 tips)
  - Bowling strategy (3 tips)
  - Team composition (3 tips)
  - Key factors

## 🎨 Design System

### Colors
- **Primary:** Green (#10b981) - Success, Grass
- **Secondary:** Blue (#3b82f6) - Info, Weather
- **Warning:** Orange (#f59e0b) - Alerts
- **Error:** Red (#ef4444) - Critical

### Pitch Type Colors
- **Batting Friendly:** Green (#10b981)
- **Bowling Friendly:** Blue (#3b82f6)
- **Spin Friendly:** Orange (#f59e0b)
- **Seam Friendly:** Red (#ef4444)

## 📂 Project Structure

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Navigation header
│   │   ├── Header.css
│   │   ├── UploadSection.jsx   # Image upload & options
│   │   ├── UploadSection.css
│   │   ├── ResultsSection.jsx  # Analysis results
│   │   ├── ResultsSection.css
│   │   ├── Footer.jsx          # Footer with links
│   │   └── Footer.css
│   ├── App.jsx                 # Main app component
│   ├── App.css
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## 🔧 Configuration

### API URL

Update in `UploadSection.jsx`:
```javascript
const API_URL = 'http://localhost:8000'
```

For production:
```javascript
const API_URL = 'https://your-api-url.com'
```

## 📱 Responsive Breakpoints

- **Desktop:** 1024px+
- **Tablet:** 768px - 1023px
- **Mobile:** < 768px

## 🎯 API Integration

The frontend calls these backend endpoints:

### POST /api/analyze
Complete analysis with all features
```javascript
{
  image: File,
  latitude: number,
  longitude: number,
  city: string,
  include_weather: boolean
}
```

### POST /api/quick-analyze
Fast classification only
```javascript
{
  image: File
}
```

## 🐛 Troubleshooting

### Backend not connecting
1. Ensure backend is running: `python backend/app.py`
2. Check API URL in `UploadSection.jsx`
3. Verify CORS is enabled in backend

### Build errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port already in use
```bash
# Change port in vite.config.js
server: {
  port: 3001  // Change from 3000
}
```

## 🚀 Deployment

### Vercel (Recommended)

1. Push to GitHub
2. Go to https://vercel.com
3. Import repository
4. Set build command: `npm run build`
5. Set output directory: `dist`
6. Add environment variable: `VITE_API_URL=your-backend-url`
7. Deploy!

### Netlify

1. Push to GitHub
2. Go to https://netlify.com
3. New site from Git
4. Build command: `npm run build`
5. Publish directory: `dist`
6. Deploy!

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:
```bash
docker build -t pitch-insight-frontend .
docker run -p 80:80 pitch-insight-frontend
```

## 📄 License

MIT License - See LICENSE file for details

---

**🎉 Your frontend is ready!**

Start the dev server and enjoy the beautiful UI! 🚀
