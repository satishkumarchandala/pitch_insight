"""
Advanced Weather Forecast and Cricket Impact Analysis Module
Provides comprehensive weather analysis for cricket matches including forecasts,
historical trends, and cricket-specific impact predictions.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config import WEATHER_API_KEY


class WeatherForecastAnalyzer:
    """Analyze weather forecasts and their impact on cricket matches"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1"
    
    def get_comprehensive_forecast(
        self, 
        location: str, 
        match_format: str = "odi",
        match_start_time: Optional[str] = None
    ) -> Dict:
        """
        Get comprehensive weather forecast with cricket-specific analysis
        
        Args:
            location: City name or coordinates (lat,lon)
            match_format: "test", "odi", or "t20"
            match_start_time: Match start time (HH:MM format)
            
        Returns:
            Complete weather forecast with cricket impact analysis
        """
        if not self.api_key or self.api_key == "your-weather-api-key-here":
            return {"error": "Weather API key not configured"}
        
        try:
            # Normalize match format to lowercase
            match_format = match_format.lower().strip()
            
            # Validate match format
            if match_format not in ["test", "odi", "t20"]:
                return {"error": f"Invalid match format: {match_format}. Must be 'test', 'odi', or 't20'"}
            
            # Determine forecast days based on match format
            # Test matches: 5 days with session-wise analysis
            # ODI/T20: 1-2 days with innings-wise analysis
            forecast_days = 5 if match_format == "test" else 2
            
            # Fetch forecast data
            forecast_url = f"{self.base_url}/forecast.json"
            forecast_response = requests.get(
                forecast_url,
                params={
                    "key": self.api_key,
                    "q": location,
                    "days": forecast_days,
                    "aqi": "no",
                    "alerts": "yes"
                },
                timeout=10
            )
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            # Fetch historical data (last 3 days)
            historical_data = self._fetch_historical_weather(location)
            
            # Parse and analyze
            current_weather = self._parse_current_weather(forecast_data)
            historical_analysis = self._analyze_historical_trends(historical_data)
            
            # Branch based on match format
            if match_format == "test":
                print(f"✅ Analyzing Test Match - 5 days with session-wise breakdown")
                analysis = self._analyze_test_match_forecast(
                    forecast_data, 
                    current_weather, 
                    historical_analysis
                )
            else:  # ODI or T20
                print(f"✅ Analyzing {match_format.upper()} Match - innings-wise breakdown with dew analysis")
                analysis = self._analyze_limited_overs_forecast(
                    forecast_data,
                    current_weather,
                    historical_analysis,
                    match_format,
                    match_start_time
                )
            
            return {
                "success": True,
                "match_format": match_format,
                "location": f"{forecast_data['location']['name']}, {forecast_data['location']['country']}",
                "current": current_weather,
                "historical": historical_analysis,
                **analysis
            }
            
        except requests.RequestException as e:
            print(f"❌ Weather API error: {str(e)}")
            return {"error": f"Failed to fetch weather: {str(e)}"}
        except Exception as e:
            print(f"❌ Weather analysis error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _fetch_historical_weather(self, location: str) -> Dict:
        """Fetch historical weather for past 3 days"""
        try:
            history_url = f"{self.base_url}/history.json"
            
            # Get last 3 days
            historical_data = []
            for days_ago in range(1, 4):
                date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                response = requests.get(
                    history_url,
                    params={"key": self.api_key, "q": location, "dt": date},
                    timeout=5
                )
                if response.status_code == 200:
                    historical_data.append(response.json())
            
            return {"days": historical_data}
        except Exception as e:
            print(f"⚠️ Historical weather fetch failed: {str(e)}")
            return {"days": []}
    
    def _parse_current_weather(self, data: Dict) -> Dict:
        """Parse current weather conditions"""
        current = data.get("current", {})
        location = data.get("location", {})
        
        return {
            "temperature": current.get("temp_c", 0),
            "feels_like": current.get("feelslike_c", 0),
            "humidity": current.get("humidity", 0),
            "dew_point": current.get("dewpoint_c", 0),
            "uv_index": current.get("uv", 0),
            "wind_speed": current.get("wind_kph", 0),
            "wind_direction": current.get("wind_dir", "N"),
            "wind_degree": current.get("wind_degree", 0),
            "cloud_cover": current.get("cloud", 0),
            "pressure": current.get("pressure_mb", 0),
            "visibility": current.get("vis_km", 0),
            "rainfall": current.get("precip_mm", 0),
            "conditions": current.get("condition", {}).get("text", "Unknown"),
            "is_day": current.get("is_day", 1)
        }
    
    def _analyze_historical_trends(self, historical_data: Dict) -> Dict:
        """Analyze historical weather trends"""
        days = historical_data.get("days", [])
        
        if not days:
            return {
                "rainfall_24h": 0,
                "rainfall_48h": 0,
                "rainfall_72h": 0,
                "avg_temp_3d": 0,
                "avg_temp_7d": 0,
                "recent_conditions": "No historical data",
                "pitch_moisture_inference": "Unknown",
                "surface_hardness_inference": "Unknown",
                "crack_potential": "Unknown",
                "interpretation": "Insufficient historical data"
            }
        
        # Calculate rainfall totals
        rainfall_totals = []
        temps = []
        
        for day in days:
            day_data = day.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
            rainfall_totals.append(day_data.get("totalprecip_mm", 0))
            temps.append(day_data.get("avgtemp_c", 0))
        
        total_rain_72h = sum(rainfall_totals)
        total_rain_48h = sum(rainfall_totals[:2]) if len(rainfall_totals) >= 2 else 0
        total_rain_24h = rainfall_totals[0] if rainfall_totals else 0
        
        avg_temp = sum(temps) / len(temps) if temps else 0
        
        # Infer pitch conditions
        if total_rain_72h > 20:
            moisture_inference = "Very High - Heavy rain in past 72h"
            surface_hardness = "Soft - Significant moisture retention"
            crack_potential = "Low - Surface too wet for cracks"
            interpretation = "Pitch likely to be soft with high moisture. Expect significant swing and seam movement. Slow outfield. Batting will be challenging early on."
        elif total_rain_72h > 10:
            moisture_inference = "High - Moderate rain in past 72h"
            surface_hardness = "Moderate - Some moisture present"
            crack_potential = "Low to Medium"
            interpretation = "Pitch has moderate moisture. Early swing and seam movement expected. Conditions may favor bowlers in first session."
        elif total_rain_72h > 2:
            moisture_inference = "Moderate - Light rain recently"
            surface_hardness = "Firm - Minimal moisture"
            crack_potential = "Medium"
            interpretation = "Balanced conditions. Some early moisture but pitch should firm up quickly. Good batting conditions after initial period."
        elif avg_temp > 30:
            moisture_inference = "Low - Hot and dry conditions"
            surface_hardness = "Hard - Baked surface"
            crack_potential = "High - Dry heat accelerates cracking"
            interpretation = "Pitch has been baking in heat. Expect a hard, dry surface with potential cracks. Spin will come into play. Reverse swing possible."
        else:
            moisture_inference = "Low to Moderate - Dry conditions"
            surface_hardness = "Firm to Hard"
            crack_potential = "Medium to High"
            interpretation = "Dry pitch with good batting conditions. Pace and bounce expected. Surface may deteriorate later."
        
        recent_conditions = "Wet" if total_rain_72h > 5 else "Dry" if total_rain_72h == 0 else "Mixed"
        
        return {
            "rainfall_24h": round(total_rain_24h, 1),
            "rainfall_48h": round(total_rain_48h, 1),
            "rainfall_72h": round(total_rain_72h, 1),
            "avg_temp_3d": round(avg_temp, 1),
            "avg_temp_7d": round(avg_temp, 1),  # Using 3-day average as proxy
            "recent_conditions": recent_conditions,
            "pitch_moisture_inference": moisture_inference,
            "surface_hardness_inference": surface_hardness,
            "crack_potential": crack_potential,
            "interpretation": interpretation
        }
    
    def _calculate_cricket_impact(
        self, 
        temp: float, 
        humidity: float, 
        cloud: float, 
        rain_chance: int,
        wind_speed: float,
        is_night: bool = False
    ) -> Dict:
        """Calculate cricket-specific weather impact"""
        # Swing potential calculation
        swing_score = 0
        if cloud > 70:
            swing_score += 30
        elif cloud > 50:
            swing_score += 20
        elif cloud > 30:
            swing_score += 10
        
        if humidity > 75:
            swing_score += 25
        elif humidity > 60:
            swing_score += 15
        elif humidity > 45:
            swing_score += 5
        
        if 15 <= temp <= 25:
            swing_score += 20
        elif 10 <= temp <= 30:
            swing_score += 10
        
        if wind_speed < 15:
            swing_score += 15
        elif wind_speed < 25:
            swing_score += 5
        
        if rain_chance > 30:
            swing_score += 10
        
        # Categorize swing potential
        if swing_score >= 70:
            swing_potential = "Very High"
        elif swing_score >= 50:
            swing_potential = "High"
        elif swing_score >= 30:
            swing_potential = "Medium"
        else:
            swing_potential = "Low"
        
        # Seam movement
        if cloud > 60 and humidity > 60:
            seam_movement = "Significant"
        elif cloud > 40 or humidity > 50:
            seam_movement = "Moderate"
        else:
            seam_movement = "Minimal"
        
        # Spin assistance
        spin_score = 0
        if temp > 30:
            spin_score += 30
        elif temp > 25:
            spin_score += 15
        
        if humidity < 50:
            spin_score += 20
        elif humidity < 65:
            spin_score += 10
        
        if cloud < 40:
            spin_score += 20
        elif cloud < 60:
            spin_score += 10
        
        if spin_score >= 50:
            spin_assistance = "High"
        elif spin_score >= 30:
            spin_assistance = "Moderate"
        else:
            spin_assistance = "Low"
        
        # Dew likelihood (for night matches)
        if is_night:
            dew_gap = temp - (humidity / 5)  # Simplified dew point approximation
            if humidity > 75 and temp < 25:
                dew_likelihood = "High"
            elif humidity > 65:
                dew_likelihood = "Moderate"
            else:
                dew_likelihood = "Low"
        else:
            dew_likelihood = "Not Applicable (Day Match)"
            dew_gap = 0
        
        # Pitch moisture level
        if rain_chance > 50 or humidity > 80:
            pitch_moisture = "High"
        elif rain_chance > 20 or humidity > 65:
            pitch_moisture = "Moderate"
        else:
            pitch_moisture = "Low"
        
        # Bowling vs Batting advantage
        bowling_advantage = int((swing_score * 0.6 + (100 - temp if temp > 35 else 50)) / 1.5)
        batting_advantage = 100 - bowling_advantage
        
        # Recommended strategy
        if bowling_advantage > 65:
            strategy = "Bowl first - Conditions heavily favor bowlers"
        elif bowling_advantage > 55:
            strategy = "Consider bowling first - Bowlers have advantage"
        elif batting_advantage > 55:
            strategy = "Bat first - Good batting conditions"
        else:
            strategy = "Balanced conditions - Toss less critical"
        
        # Key factors
        factors = []
        if swing_score > 50:
            factors.append(f"High swing potential ({swing_potential})")
        if seam_movement == "Significant":
            factors.append("Significant seam movement expected")
        if spin_score > 40:
            factors.append(f"Good spin assistance ({spin_assistance})")
        if rain_chance > 30:
            factors.append(f"{rain_chance}% chance of rain - possible interruptions")
        if dew_likelihood == "High":
            factors.append("High dew factor - bowling second challenging")
        if temp > 35:
            factors.append(f"Very hot ({temp}°C) - player fatigue factor")
        
        return {
            "swing_potential": swing_potential,
            "swing_score": round(swing_score, 1),
            "seam_movement": seam_movement,
            "spin_assistance": spin_assistance,
            "spin_score": round(spin_score, 1),
            "pitch_moisture_level": pitch_moisture,
            "bowling_advantage": bowling_advantage,
            "batting_advantage": batting_advantage,
            "dew_likelihood": dew_likelihood,
            "recommended_strategy": strategy,
            "key_factors": factors
        }
    
    def _analyze_test_match_forecast(
        self, 
        forecast_data: Dict,
        current: Dict,
        historical: Dict
    ) -> Dict:
        """Analyze 5-day Test match forecast with session-wise breakdown"""
        forecast_days = forecast_data.get("forecast", {}).get("forecastday", [])
        
        daily_forecasts = []
        pitch_behavior_trend = []
        key_risks = []
        phase_advantages = {}
        
        for day_num, day in enumerate(forecast_days[:5], 1):
            day_data = day.get("day", {})
            hours = day.get("hour", [])
            
            # Define session times (assuming 10am-6pm match time)
            # Session 1: 10am-12:30pm, Lunch, Session 2: 1:30pm-3:30pm, Tea, Session 3: 4pm-6pm
            session_1_hours = hours[10:13]  # 10am-12pm
            session_2_hours = hours[13:16]  # 1pm-3pm
            session_3_hours = hours[16:18]  # 4pm-6pm
            
            sessions = []
            
            # Analyze each session
            for session_num, (session_hours, session_name, time_range) in enumerate([
                (session_1_hours, "1st Session (Morning)", "10:00 AM - 12:30 PM"),
                (session_2_hours, "2nd Session (Afternoon)", "1:30 PM - 3:30 PM"),
                (session_3_hours, "3rd Session (Evening)", "4:00 PM - 6:00 PM")
            ], 1):
                if not session_hours:
                    continue
                
                # Calculate session averages
                avg_temp = sum(h.get("temp_c", 0) for h in session_hours) / len(session_hours)
                avg_humidity = sum(h.get("humidity", 0) for h in session_hours) / len(session_hours)
                avg_cloud = sum(h.get("cloud", 0) for h in session_hours) / len(session_hours)
                avg_wind = sum(h.get("wind_kph", 0) for h in session_hours) / len(session_hours)
                total_rain = sum(h.get("precip_mm", 0) for h in session_hours)
                max_rain_chance = max(h.get("chance_of_rain", 0) for h in session_hours)
                
                conditions = session_hours[0].get("condition", {}).get("text", "Unknown")
                
                # Calculate cricket impact
                is_night = session_num == 3 and avg_temp < 25
                impact = self._calculate_cricket_impact(
                    avg_temp, avg_humidity, avg_cloud, max_rain_chance, avg_wind, is_night
                )
                
                session_data = {
                    "session_name": f"Day {day_num} - {session_name}",
                    "time_range": time_range,
                    "avg_temperature": round(avg_temp, 1),
                    "avg_humidity": round(avg_humidity, 1),
                    "avg_cloud_cover": round(avg_cloud, 1),
                    "total_rainfall": round(total_rain, 1),
                    "chance_of_rain": max_rain_chance,
                    "avg_wind_speed": round(avg_wind, 1),
                    "conditions_summary": conditions,
                    **impact
                }
                
                sessions.append(session_data)
            
            # Daily analysis
            daily_rain = day_data.get("totalprecip_mm", 0)
            if daily_rain > 10:
                key_risks.append(f"Day {day_num}: High rain risk ({daily_rain}mm expected)")
            
            # Pitch deterioration analysis
            if day_num == 1:
                deterioration = "Fresh - Minimal wear"
                crack_dev = "None - Pitch intact"
                outfield = "Good - Well maintained"
            elif day_num == 2:
                deterioration = "Light - Some footmarks"
                crack_dev = "Minimal - Surface holding up"
                outfield = "Good"
            elif day_num == 3:
                deterioration = "Moderate - Visible wear"
                crack_dev = "Developing - Small cracks appearing"
                outfield = "Fair"
            elif day_num == 4:
                deterioration = "Significant - Clear footmarks and worn areas"
                crack_dev = "Moderate - Cracks widening"
                outfield = "Fair to Worn"
            else:
                deterioration = "Heavy - Extensive wear"
                crack_dev = "Significant - Large cracks, uneven bounce"
                outfield = "Worn"
            
            # Overall advantage
            avg_bowling_adv = sum(s["bowling_advantage"] for s in sessions) / len(sessions) if sessions else 50
            if avg_bowling_adv > 60:
                overall_adv = "Bowlers"
            elif avg_bowling_adv < 40:
                overall_adv = "Batters"
            else:
                overall_adv = "Balanced"
            
            daily_forecasts.append({
                "date": day.get("date"),
                "day_number": day_num,
                "max_temp": day_data.get("maxtemp_c", 0),
                "min_temp": day_data.get("mintemp_c", 0),
                "avg_humidity": day_data.get("avghumidity", 0),
                "total_rainfall": daily_rain,
                "chance_of_rain": day_data.get("daily_chance_of_rain", 0),
                "sunrise": day.get("astro", {}).get("sunrise", ""),
                "sunset": day.get("astro", {}).get("sunset", ""),
                "uv_index": day_data.get("uv", 0),
                "conditions_summary": day_data.get("condition", {}).get("text", ""),
                "sessions": sessions,
                "pitch_deterioration_rate": deterioration,
                "crack_development": crack_dev,
                "outfield_condition": outfield,
                "overall_advantage": overall_adv
            })
            
            phase_advantages[f"Day {day_num}"] = overall_adv
            pitch_behavior_trend.append(f"Day {day_num}: {crack_dev}, advantage {overall_adv}")
        
        # Generate match summary
        total_rain = sum(d["total_rainfall"] for d in daily_forecasts)
        
        if total_rain > 30:
            match_summary = f"High rain risk across the Test match ({total_rain}mm total forecast). Expect interruptions and favorable bowling conditions when play is possible. Pitch may not deteriorate normally due to moisture. Seam and swing will dominate."
        elif total_rain > 10:
            match_summary = f"Moderate rain expected ({total_rain}mm total). Some interruptions likely. Good bowling conditions early, pitch should deteriorate normally from Day 3 onwards. Spin will come into play on Days 4-5."
        else:
            match_summary = f"Dry conditions expected ({total_rain}mm rain). Pitch will deteriorate progressively. Early advantage to pace bowlers, increasing spin assistance from Day 3. Hard, bouncy surface. Good batting conditions in first two days."
        
        if not key_risks:
            key_risks.append("Minimal weather-related risks")
        
        recommendations = [
            "Monitor cloud cover for swing conditions",
            "Pace bowlers crucial in first 2 days",
            "Expect increasing spin from Day 3 onwards",
            "Bat first if winning toss (unless heavy cloud/rain forecast)",
            "Plan for pitch to favor bowlers progressively"
        ]
        
        return {
            "daily_forecasts": daily_forecasts,
            "hourly_forecast": self._extract_hourly_forecast(forecast_days),
            "pitch_behavior_trend": " | ".join(pitch_behavior_trend[:5]),
            "key_risks": key_risks,
            "phase_wise_advantage": phase_advantages,
            "match_condition_summary": match_summary,
            "recommendations": recommendations
        }
    
    def _analyze_limited_overs_forecast(
        self,
        forecast_data: Dict,
        current: Dict,
        historical: Dict,
        match_format: str,
        match_start_time: Optional[str] = None
    ) -> Dict:
        """Analyze limited-overs match forecast (ODI/T20) with innings-wise breakdown"""
        forecast_days = forecast_data.get("forecast", {}).get("forecastday", [])
        
        if not forecast_days:
            return {}
        
        # Get today's hourly forecast
        today = forecast_days[0]
        hours = today.get("hour", [])
        
        # Determine match hours
        if match_format == "t20":
            match_duration = 4  # 4 hours for T20
            default_start = 19  # 7 PM
        else:  # ODI
            match_duration = 8  # 8 hours for ODI
            default_start = 10  # 10 AM
        
        if match_start_time:
            try:
                start_hour = int(match_start_time.split(":")[0])
            except:
                start_hour = default_start
        else:
            start_hour = default_start
        
        # Split into innings
        innings_1_hours = hours[start_hour:start_hour + (match_duration // 2)]
        innings_2_hours = hours[start_hour + (match_duration // 2):start_hour + match_duration]
        
        innings_forecasts = []
        
        for innings_num, (innings_hours, innings_name) in enumerate([
            (innings_1_hours, "1st Innings"),
            (innings_2_hours, "2nd Innings")
        ], 1):
            if not innings_hours:
                continue
            
            # Calculate innings averages
            avg_temp = sum(h.get("temp_c", 0) for h in innings_hours) / len(innings_hours)
            avg_humidity = sum(h.get("humidity", 0) for h in innings_hours) / len(innings_hours)
            avg_cloud = sum(h.get("cloud", 0) for h in innings_hours) / len(innings_hours)
            avg_wind = sum(h.get("wind_kph", 0) for h in innings_hours) / len(innings_hours)
            total_rain = sum(h.get("precip_mm", 0) for h in innings_hours)
            max_rain_chance = max(h.get("chance_of_rain", 0) for h in innings_hours)
            
            first_hour = innings_hours[0]
            last_hour = innings_hours[-1]
            time_range = f"{first_hour.get('time', '').split()[1]} - {last_hour.get('time', '').split()[1]}"
            
            is_night = first_hour.get("is_day", 1) == 0 or innings_num == 2
            
            # Calculate cricket impact
            impact = self._calculate_cricket_impact(
                avg_temp, avg_humidity, avg_cloud, max_rain_chance, avg_wind, is_night
            )
            
            # Additional innings-specific factors
            if innings_num == 2 and is_night:
                impact["key_factors"].append("Dew factor - difficult for bowlers")
                if match_format == "t20":
                    impact["recommended_strategy"] = "Chase - dew and short boundaries favor batting second"
            
            innings_data = {
                "session_name": innings_name,
                "time_range": time_range,
                "avg_temperature": round(avg_temp, 1),
                "avg_humidity": round(avg_humidity, 1),
                "avg_cloud_cover": round(avg_cloud, 1),
                "total_rainfall": round(total_rain, 1),
                "chance_of_rain": max_rain_chance,
                "avg_wind_speed": round(avg_wind, 1),
                "conditions_summary": first_hour.get("condition", {}).get("text", "Unknown"),
                **impact
            }
            
            innings_forecasts.append(innings_data)
        
        # Generate match summary
        key_risks = []
        
        total_rain = sum(i["total_rainfall"] for i in innings_forecasts)
        if total_rain > 5:
            key_risks.append(f"Rain risk: {total_rain}mm expected - DLS possible")
        
        if innings_forecasts[0]["swing_score"] > 60:
            key_risks.append("Early swing conditions - challenging for openers")
        
        if len(innings_forecasts) > 1:
            if innings_forecasts[1]["dew_likelihood"] == "High":
                key_risks.append("High dew in 2nd innings - advantage to chasing team")
        
        # Pitch behavior
        if historical["recent_conditions"] == "Wet":
            pitch_behavior = "Slow and low - pitch moisture present. Batting difficult initially. Spin effective."
        elif historical["recent_conditions"] == "Dry":
            pitch_behavior = "Hard and bouncy - good pace and carry. Batting friendly. Some reverse swing possible later."
        else:
            pitch_behavior = "Balanced surface - good for both batting and bowling. Conditions may favor team batting first."
        
        # Phase-wise advantage
        phase_advantages = {
            innings_forecasts[0]["session_name"]: "Bowlers" if innings_forecasts[0]["bowling_advantage"] > 55 else "Batters",
        }
        if len(innings_forecasts) > 1:
            phase_advantages[innings_forecasts[1]["session_name"]] = "Bowlers" if innings_forecasts[1]["bowling_advantage"] > 55 else "Batters"
        
        # Match summary
        if match_format == "t20":
            match_summary = f"T20 Match: {pitch_behavior} {'Chase favorable due to dew and short boundaries.' if any(i.get('dew_likelihood') == 'High' for i in innings_forecasts) else 'Balanced conditions.'} Target: 160-180 par score."
        else:
            match_summary = f"ODI Match: {pitch_behavior} First innings: {innings_forecasts[0]['batting_advantage']}% batting advantage. Second innings: {'dew factor critical' if any(i.get('dew_likelihood') == 'High' for i in innings_forecasts) else 'similar conditions'}. Target: 270-300 competitive."
        
        recommendations = [
            "Monitor powerplay conditions closely",
            "Use pace bowlers in favorable swing conditions",
            "Middle overs crucial for building partnerships",
            f"{'Bat second if possible - dew advantage' if any(i.get('dew_likelihood') == 'High' for i in innings_forecasts) else 'Bat first to set target'}",
            "Stay alert for weather interruptions"
        ]
        
        if not key_risks:
            key_risks.append("Minimal weather-related risks")
        
        return {
            "innings_forecasts": innings_forecasts,
            "hourly_forecast": self._extract_hourly_forecast(forecast_days),
            "pitch_behavior_trend": pitch_behavior,
            "key_risks": key_risks,
            "phase_wise_advantage": phase_advantages,
            "match_condition_summary": match_summary,
            "recommendations": recommendations
        }
    
    def _extract_hourly_forecast(self, forecast_days: List[Dict]) -> List[Dict]:
        """Extract hourly forecast for next 24 hours"""
        hourly = []
        
        for day in forecast_days[:2]:  # First 2 days for 48h coverage
            for hour in day.get("hour", []):
                hourly.append({
                    "time": hour.get("time", ""),
                    "temperature": hour.get("temp_c", 0),
                    "humidity": hour.get("humidity", 0),
                    "cloud_cover": hour.get("cloud", 0),
                    "chance_of_rain": hour.get("chance_of_rain", 0),
                    "rainfall": hour.get("precip_mm", 0),
                    "wind_speed": hour.get("wind_kph", 0),
                    "wind_direction": hour.get("wind_dir", "N"),
                    "conditions": hour.get("condition", {}).get("text", ""),
                    "will_it_rain": hour.get("will_it_rain", 0) == 1
                })
        
        return hourly[:24]  # Return 24 hours


# Singleton instance
_weather_analyzer = None

def get_weather_analyzer() -> WeatherForecastAnalyzer:
    """Get singleton weather analyzer instance"""
    global _weather_analyzer
    if _weather_analyzer is None:
        _weather_analyzer = WeatherForecastAnalyzer(WEATHER_API_KEY)
    return _weather_analyzer
