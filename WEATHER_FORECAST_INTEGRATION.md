# Weather Forecast Integration Documentation

## Overview

The Pitch Insight platform now includes a comprehensive weather forecast and cricket-impact analysis system that goes beyond simple current conditions. This advanced system provides:

- **5-day forecasts for Test matches** with session-wise breakdown
- **Innings-wise analysis for limited-overs formats** (ODI/T20)
- **Historical weather trend analysis** (past 3 days)
- **Cricket-specific impact predictions** (swing, seam, spin, dew)
- **Match strategy recommendations** based on forecasted conditions

---

## API Endpoints

### 1. Current Weather (Basic)

**Endpoint:** `GET /api/weather`

**Description:** Returns only current weather conditions

**Query Parameters:**
- `city` (string, optional): City name
- `latitude` (float, optional): Location latitude
- `longitude` (float, optional): Location longitude

**Response:**
```json
{
  "success": true,
  "location": {
    "name": "Mumbai",
    "region": "Maharashtra",
    "country": "India"
  },
  "current": {
    "temperature": 28,
    "humidity": 75,
    "conditions": "Partly cloudy"
  }
}
```

---

### 2. Comprehensive Weather Forecast (Advanced) ⭐

**Endpoint:** `GET /api/weather/forecast`

**Description:** Returns comprehensive weather forecast with cricket-specific analysis

**Query Parameters:**
- `city` (string, optional): City name
- `latitude` (float, optional): Location latitude  
- `longitude` (float, optional): Location longitude
- `match_format` (string, required): Match format - `test`, `odi`, or `t20`
- `match_start_time` (string, optional): Match start time in HH:MM format (e.g., "10:00", "19:00")

**Example Request:**
```bash
GET /api/weather/forecast?city=Mumbai&match_format=test

GET /api/weather/forecast?city=London&match_format=t20&match_start_time=19:00

GET /api/weather/forecast?latitude=28.6139&longitude=77.2090&match_format=odi
```

**Response Structure:**

#### For Test Matches:
```json
{
  "success": true,
  "match_format": "test",
  "location": "Mumbai, India",
  "current": {
    "temperature": 28.5,
    "humidity": 75,
    "cloud_cover": 45,
    "conditions": "Partly cloudy",
    "wind_speed": 12
  },
  "historical": {
    "rainfall_72h": 5.2,
    "avg_temp_3d": 27.8,
    "recent_conditions": "Mixed",
    "pitch_moisture_inference": "Moderate - Light rain recently",
    "surface_hardness_inference": "Firm - Minimal moisture",
    "crack_potential": "Medium",
    "interpretation": "Balanced conditions. Some early moisture but pitch should firm up quickly."
  },
  "daily_forecasts": [
    {
      "date": "2026-01-01",
      "day_number": 1,
      "max_temp": 30,
      "min_temp": 22,
      "total_rainfall": 0,
      "chance_of_rain": 10,
      "conditions_summary": "Sunny",
      "sessions": [
        {
          "session_name": "Day 1 - 1st Session (Morning)",
          "time_range": "10:00 AM - 12:30 PM",
          "avg_temperature": 26.5,
          "avg_humidity": 68,
          "avg_cloud_cover": 30,
          "swing_potential": "Medium",
          "swing_score": 45.0,
          "seam_movement": "Moderate",
          "spin_assistance": "Low",
          "spin_score": 25.0,
          "pitch_moisture_level": "Moderate",
          "bowling_advantage": 58,
          "batting_advantage": 42,
          "dew_likelihood": "Not Applicable (Day Match)",
          "recommended_strategy": "Consider bowling first - Bowlers have advantage",
          "key_factors": [
            "High swing potential (Medium)",
            "Significant seam movement expected"
          ]
        },
        {
          "session_name": "Day 1 - 2nd Session (Afternoon)",
          "time_range": "1:30 PM - 3:30 PM",
          "avg_temperature": 29.2,
          "swing_potential": "Low",
          "batting_advantage": 55,
          "recommended_strategy": "Balanced conditions - Toss less critical"
        },
        {
          "session_name": "Day 1 - 3rd Session (Evening)",
          "time_range": "4:00 PM - 6:00 PM",
          "avg_temperature": 27.8,
          "swing_potential": "Medium",
          "bowling_advantage": 52
        }
      ],
      "pitch_deterioration_rate": "Fresh - Minimal wear",
      "crack_development": "None - Pitch intact",
      "outfield_condition": "Good - Well maintained",
      "overall_advantage": "Bowlers"
    },
    {
      "date": "2026-01-02",
      "day_number": 2,
      "pitch_deterioration_rate": "Light - Some footmarks",
      "overall_advantage": "Balanced"
    },
    {
      "date": "2026-01-03",
      "day_number": 3,
      "pitch_deterioration_rate": "Moderate - Visible wear",
      "crack_development": "Developing - Small cracks appearing",
      "overall_advantage": "Balanced"
    },
    {
      "date": "2026-01-04",
      "day_number": 4,
      "crack_development": "Moderate - Cracks widening",
      "overall_advantage": "Bowlers"
    },
    {
      "date": "2026-01-05",
      "day_number": 5,
      "pitch_deterioration_rate": "Heavy - Extensive wear",
      "crack_development": "Significant - Large cracks, uneven bounce",
      "overall_advantage": "Bowlers"
    }
  ],
  "pitch_behavior_trend": "Day 1: None - Pitch intact, advantage Bowlers | Day 2: Minimal - Surface holding up, advantage Balanced | Day 3: Developing - Small cracks appearing, advantage Balanced | Day 4: Moderate - Cracks widening, advantage Bowlers | Day 5: Significant - Large cracks, uneven bounce, advantage Bowlers",
  "key_risks": [
    "Minimal weather-related risks"
  ],
  "phase_wise_advantage": {
    "Day 1": "Bowlers",
    "Day 2": "Balanced",
    "Day 3": "Balanced",
    "Day 4": "Bowlers",
    "Day 5": "Bowlers"
  },
  "match_condition_summary": "Dry conditions expected (5.2mm rain). Pitch will deteriorate progressively. Early advantage to pace bowlers, increasing spin assistance from Day 3. Hard, bouncy surface. Good batting conditions in first two days.",
  "recommendations": [
    "Monitor cloud cover for swing conditions",
    "Pace bowlers crucial in first 2 days",
    "Expect increasing spin from Day 3 onwards",
    "Bat first if winning toss (unless heavy cloud/rain forecast)",
    "Plan for pitch to favor bowlers progressively"
  ]
}
```

#### For Limited-Overs Matches (ODI/T20):
```json
{
  "success": true,
  "match_format": "t20",
  "location": "Mumbai, India",
  "current": { ... },
  "historical": { ... },
  "innings_forecasts": [
    {
      "session_name": "1st Innings",
      "time_range": "19:00 - 21:00",
      "avg_temperature": 26.0,
      "avg_humidity": 78,
      "swing_potential": "High",
      "swing_score": 65.0,
      "seam_movement": "Significant",
      "spin_assistance": "Low",
      "dew_likelihood": "Not Applicable (Day Match)",
      "bowling_advantage": 62,
      "batting_advantage": 38,
      "recommended_strategy": "Bowl first - Conditions heavily favor bowlers",
      "key_factors": [
        "High swing potential (High)",
        "Significant seam movement expected"
      ]
    },
    {
      "session_name": "2nd Innings",
      "time_range": "21:30 - 23:30",
      "avg_temperature": 24.5,
      "avg_humidity": 82,
      "dew_likelihood": "High",
      "bowling_advantage": 45,
      "batting_advantage": 55,
      "recommended_strategy": "Balanced conditions - Toss less critical",
      "key_factors": [
        "High dew factor - bowling second challenging",
        "Good spin assistance (Moderate)"
      ]
    }
  ],
  "pitch_behavior_trend": "Hard and bouncy - good pace and carry. Batting friendly. Some reverse swing possible later.",
  "key_risks": [
    "High dew in 2nd innings - advantage to chasing team"
  ],
  "phase_wise_advantage": {
    "1st Innings": "Bowlers",
    "2nd Innings": "Batters"
  },
  "match_condition_summary": "T20 Match: Hard and bouncy - good pace and carry. Batting friendly. Some reverse swing possible later. Chase favorable due to dew and short boundaries. Target: 160-180 par score.",
  "recommendations": [
    "Monitor powerplay conditions closely",
    "Use pace bowlers in favorable swing conditions",
    "Middle overs crucial for building partnerships",
    "Bat second if possible - dew advantage",
    "Stay alert for weather interruptions"
  ]
}
```

---

### 3. Pitch Analysis with Forecast Integration

**Endpoint:** `POST /api/analyze`

**Description:** Complete pitch analysis with optional comprehensive weather forecast

**Form Parameters:**
- `file` (file, required): Pitch image
- `city` (string, optional): City name
- `latitude` (float, optional): Location latitude
- `longitude` (float, optional): Location longitude
- `include_weather` (boolean, default: true): Include weather data
- **`use_forecast` (boolean, default: false): Use comprehensive forecast** ⭐ NEW
- `match_type` (string, default: "odi"): Match format
- `match_start_time` (string, optional): Match start time (HH:MM)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@pitch.jpg" \
  -F "city=Mumbai" \
  -F "match_type=test" \
  -F "use_forecast=true" \
  -F "include_weather=true"
```

**Response:** Full pitch analysis + comprehensive weather forecast

---

## Cricket Impact Analysis

### Swing Potential
**Factors Considered:**
- Cloud cover (>70% = High swing)
- Humidity (>75% = High swing)
- Temperature (15-25°C = Optimal)
- Wind speed (<15 kph = Better)
- Rain chance

**Categories:**
- **Very High** (70-100): Expect significant lateral movement
- **High** (50-69): Good swing conditions
- **Medium** (30-49): Moderate assistance
- **Low** (0-29): Minimal swing

### Seam Movement
Based on cloud cover and humidity:
- **Significant**: Cloudy + High humidity
- **Moderate**: Some cloud or moisture
- **Minimal**: Clear and dry

### Spin Assistance
**Factors:**
- Temperature (>30°C = High)
- Humidity (<50% = High)  
- Cloud cover (<40% = High)
- Pitch dryness

### Dew Factor
**For Night Matches:**
- High humidity (>75%) + Cool temp (<25°C) = High dew risk
- Dew makes ball slippery, favors batting in 2nd innings

---

## Historical Trend Analysis

### Pitch Moisture Inference
Based on rainfall in past 72 hours:
- **>20mm**: Very High - Heavy rain, soft pitch, slow outfield
- **10-20mm**: High - Moderate rain, some moisture retained
- **2-10mm**: Moderate - Light rain, balanced conditions
- **0-2mm**: Low - Dry conditions

**Impact on Pitch:**
- **Wet Recent History**: Favors swing/seam, slow batting, soft surface
- **Dry Recent History**: Hard surface, cracks likely, spin-friendly

---

## Use Cases

### 1. Pre-Match Planning (Test Match)
```javascript
// Get 5-day forecast with session breakdown
const forecast = await fetch(
  '/api/weather/forecast?city=Lords&match_format=test'
);

// Analyze each day's conditions
forecast.daily_forecasts.forEach(day => {
  console.log(`Day ${day.day_number}: ${day.overall_advantage}`);
  console.log(`Deterioration: ${day.crack_development}`);
});
```

### 2. Toss Decision (T20)
```javascript
// Get innings-wise forecast
const forecast = await fetch(
  '/api/weather/forecast?city=Mumbai&match_format=t20&match_start_time=19:00'
);

const firstInnings = forecast.innings_forecasts[0];
const secondInnings = forecast.innings_forecasts[1];

if (secondInnings.dew_likelihood === 'High') {
  console.log('Recommendation: Bat second - Dew advantage');
}
```

### 3. Combined Analysis
```javascript
// Pitch analysis + weather forecast
const formData = new FormData();
formData.append('file', pitchImage);
formData.append('city', 'Delhi');
formData.append('use_forecast', 'true');
formData.append('match_type', 'odi');

const analysis = await fetch('/api/analyze', {
  method: 'POST',
  body: formData
});

// Get both pitch condition and weather impact
console.log('Pitch Type:', analysis.final_classification.prediction);
console.log('Weather Summary:', analysis.weather_forecast.match_condition_summary);
```

---

## Configuration

### Required Environment Variable
```bash
WEATHER_API_KEY=your_api_key_here
```

Get a free API key from: https://www.weatherapi.com/

### API Key Limits
- **Free Tier**: 1M calls/month, 3-day forecast
- **Paid Tiers**: Extended forecasts, historical data

---

## Error Handling

### Common Errors

1. **Weather API Key Not Configured**
```json
{
  "error": "Weather forecast service not configured. Get a free API key from https://www.weatherapi.com/"
}
```

2. **Invalid Match Format**
```json
{
  "detail": "Invalid match_format. Must be one of: test, odi, t20"
}
```

3. **Location Not Found**
```json
{
  "error": "Failed to fetch weather: Location not found"
}
```

4. **API Rate Limit**
```json
{
  "error": "Weather API rate limit exceeded"
}
```

---

## Benefits

### For Test Matches
- **Day-by-day strategy planning**
- **Session-wise bowling/batting plans**
- **Pitch deterioration predictions**
- **Rain interruption risk assessment**

### For Limited-Overs
- **Toss advantage analysis**
- **Innings-wise strategy**
- **Dew factor predictions**
- **Powerplay condition insights**

### For All Formats
- **Historical context** (past 3 days weather)
- **Cricket-specific metrics** (not generic weather)
- **Actionable recommendations**
- **Risk identification**

---

## Technical Notes

- Forecasts updated hourly from WeatherAPI.com
- Historical data covers past 3 days
- Session times: 10am-12:30pm, 1:30pm-3:30pm, 4pm-6pm (Test)
- Night matches: Auto-detected based on start time
- Fallback: If forecast fails, returns current weather only

---

## Future Enhancements

- [ ] 7-day forecast for longer tours
- [ ] Radar/satellite imagery integration
- [ ] Venue-specific historical patterns
- [ ] AI-powered outcome predictions
- [ ] Real-time weather alerts during matches
