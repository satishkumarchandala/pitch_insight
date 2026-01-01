# Weather Forecast Feature - UI Guide

## How to Use the Comprehensive Weather Forecast

### Step 1: Access Analysis Page
Navigate to the Analysis page in Pitch Insight

### Step 2: Select Image
Upload a pitch image using drag-and-drop or file selector

### Step 3: Configure Options

#### Basic Weather (Current conditions only):
1. Check "Include Weather Analysis" ☑️
2. Enter location or use "Use Current Location"
3. Select match format (Test/ODI/T20)
4. Click Analyze

#### Advanced Weather Forecast (Full forecast):
1. Check "Include Weather Analysis" ☑️
2. **Check "Comprehensive Weather Forecast"** ☑️ ⭐ NEW
3. Enter location or use GPS
4. Select match format:
   - **Test Match**: Get 5-day forecast with session-wise analysis
   - **ODI/T20**: Get innings-wise forecast with dew analysis
5. For T20/ODI: Optionally enter match start time (e.g., 19:00)
6. Click Analyze

### Step 4: View Results

The analysis results will now include:

#### For Test Matches:
- **5-Day Forecast Cards**: Expandable cards for each day
- **Session Breakdown**: Morning, Afternoon, Evening sessions
- **Pitch Deterioration Timeline**: Track how pitch changes each day
- **Bowling/Batting Advantage**: Per session metrics
- **Swing, Seam, Spin Potential**: Real-time cricket impact

#### For Limited-Overs:
- **1st Innings Forecast**: Weather during batting first
- **2nd Innings Forecast**: Weather during chase
- **Dew Factor Analysis**: Night match dew predictions
- **Bowling/Batting Advantage**: Per innings
- **Recommended Strategy**: Based on weather evolution

#### Additional Insights:
- **Historical Context**: Past 3 days rainfall and temperature
- **Pitch Inference**: Moisture level, surface hardness
- **Key Risks**: Rain interruptions, extreme conditions
- **Match Summary**: Overall condition analysis
- **Recommendations**: Actionable cricket strategies

## UI Components

### Weather Forecast Display Features:

1. **Current Conditions Card**
   - Temperature, Humidity, Wind, Cloud Cover
   - Live conditions at match location

2. **Historical Context Card** (Purple gradient)
   - 72-hour rainfall totals
   - 3-day temperature average
   - Pitch moisture inference
   - Surface condition predictions

3. **Test Match - Daily Forecast Cards** (Expandable)
   - Click any day to expand sessions
   - Each session shows:
     * Weather metrics (temp, humidity, cloud, rain)
     * Swing/Seam/Spin potential
     * Bowling vs Batting advantage bars
     * Key factors and strategy

4. **Limited-Overs - Innings Cards** (Expandable)
   - Click innings to see details
   - Weather evolution during innings
   - Dew likelihood (for night matches)
   - Phase-wise advantage

5. **Match Summary Card** (Pink gradient)
   - Overall pitch behavior trend
   - Key risks and uncertainties
   - Phase-wise advantage summary
   - Cricket-specific recommendations

## Visual Indicators

### Color Codes:
- **Blue Bars**: Bowling advantage
- **Green Bars**: Batting advantage
- **Red/Orange Badges**: High swing/risk
- **Yellow Tags**: Key factors
- **Gradient Cards**: Important summaries

### Swing Potential Colors:
- 🔴 **Red**: Very High/High swing
- 🟡 **Orange**: Medium swing
- 🟢 **Green**: Low swing

### Advantage Badges:
- 🔵 **Blue**: Bowlers advantage
- 🟢 **Green**: Batters advantage
- 🟠 **Orange**: Balanced

## Example Scenarios

### Scenario 1: Test Match Planning
```
1. Upload pitch image
2. Check "Include Weather Analysis"
3. Check "Comprehensive Weather Forecast"
4. Select "Test Match"
5. Enter location: "Lord's, London"
6. Analyze

Result: See 5-day forecast with 15 sessions (3 per day)
- Day 1 Morning: High swing, bowl first
- Day 3-4: Cracks developing, spin increasing
- Day 5: Heavy deterioration, bowlers dominate
```

### Scenario 2: T20 Night Match
```
1. Upload pitch image
2. Check "Include Weather Analysis"
3. Check "Comprehensive Weather Forecast"
4. Select "T20"
5. Enter start time: "19:00"
6. Enter location: "Mumbai"
7. Analyze

Result: See innings-wise forecast
- 1st Innings (7-9 PM): Some swing, 55% bowling advantage
- 2nd Innings (9:30-11:30 PM): High dew, 65% batting advantage
- Recommendation: Chase if you win toss
```

### Scenario 3: ODI Day Match
```
1. Upload pitch image
2. Check "Include Weather Analysis"
3. Check "Comprehensive Weather Forecast"
4. Select "ODI"
5. Enter start time: "10:00" (optional)
6. Enter location: "Adelaide"
7. Analyze

Result: See 8-hour forecast split into innings
- Historical shows dry 3 days = Hard pitch
- 1st Innings: Clear, hot, good batting
- 2nd Innings: Slightly cloudy, some swing
- Strategy: Bat first, post big total
```

## Pro Tips

1. **Always enter match start time** for T20/ODI night matches to get accurate dew predictions

2. **Check historical context** - If it rained heavily in past 3 days, expect soft pitch regardless of forecast

3. **Expand all sessions/innings** to see detailed breakdown instead of just summary

4. **Look for key factors** - Yellow tags highlight the most important weather impacts

5. **Trust the strategy recommendations** - They're generated based on complex cricket impact calculations

6. **Compare phases** - See how advantage shifts between sessions/innings

7. **Note the pitch deterioration** for Test matches - Critical for Day 4-5 strategy

## Troubleshooting

### Forecast not showing?
- Ensure "Comprehensive Weather Forecast" checkbox is checked
- Verify location is entered correctly
- Check if weather API key is configured in backend

### Only basic weather showing?
- You might have "Include Weather" checked but not "Comprehensive Forecast"
- Re-check both boxes

### Sessions not expanding?
- Click on the day header to expand/collapse
- Each day card is independently expandable

### No historical data?
- Historical API might be unavailable
- Analysis still works without it

## Next Steps

After viewing the forecast:
1. Use insights for team selection
2. Plan batting order based on swing potential
3. Decide bowling rotation per session
4. Prepare for weather interruptions
5. Save analysis for future reference

## Feedback

Found issues or have suggestions? The comprehensive forecast is continuously improving!
