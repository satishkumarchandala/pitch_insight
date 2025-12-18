"""
Test script for Pitch Insight API
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("🏥 Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_get_classes():
    """Test get classes endpoint"""
    print("\n" + "="*60)
    print("📚 Testing Get Classes")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/classes")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Classes: {data['classes']}")
    
    return response.status_code == 200


def test_quick_analysis(image_path: str):
    """Test quick analysis endpoint"""
    print("\n" + "="*60)
    print("⚡ Testing Quick Analysis")
    print("="*60)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/api/quick-analyze", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Quick Analysis Results:")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Confidence: {data['confidence']:.2f}%")
        print(f"   Processing Time: {data['processing_time']:.2f}s")
        print(f"\n📊 Probabilities:")
        for cls, prob in data['probabilities'].items():
            print(f"   {cls}: {prob:.2f}%")
    else:
        print(f"❌ Error: {response.json()}")
    
    return response.status_code == 200


def test_complete_analysis(image_path: str, with_weather: bool = False):
    """Test complete analysis endpoint"""
    print("\n" + "="*60)
    print("🔍 Testing Complete Analysis")
    print("="*60)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Mumbai coordinates for testing
    data = {
        'include_weather': str(with_weather).lower()
    }
    
    if with_weather:
        data.update({
            'latitude': 28.6139,
            'longitude': 77.2090,
            'city': 'Mumbai'
        })
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n✅ Analysis ID: {result['analysis_id']}")
        print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
        
        # Pitch Detection
        print(f"\n🎯 Pitch Detection:")
        pd = result['pitch_detection']
        print(f"   Detected: {pd['detected']}")
        print(f"   Confidence: {pd['confidence']*100:.1f}%")
        
        # Features
        print(f"\n🔬 Features:")
        features = result['features']
        print(f"   Grass: {features['grass_coverage']['percentage']:.1f}% ({features['grass_coverage']['level']})")
        print(f"   Cracks: {features['cracks']['severity']} ({features['cracks']['count']} detected)")
        print(f"   Moisture: {features['moisture']['level']} (score: {features['moisture']['score']:.1f})")
        print(f"   Color: {features['color']['type']}")
        print(f"   Texture: {features['texture']['type']}")
        print(f"   Brightness: {features['brightness']['level']}")
        
        # ML Classification
        print(f"\n🤖 ML Classification:")
        ml = result['ml_classification']
        print(f"   Prediction: {ml['prediction']}")
        print(f"   Confidence: {ml['confidence']:.2f}%")
        
        # Final Classification
        print(f"\n🏆 Final Classification:")
        final = result['final_classification']
        print(f"   Prediction: {final['prediction'].upper()}")
        print(f"   Confidence: {final['confidence']:.2f}%")
        if final['adjustments']:
            print(f"   Adjustments: {', '.join(final['adjustments'])}")
            print(f"   Reasons: {', '.join(final['reasons'])}")
        
        # Weather
        if result['weather']:
            print(f"\n🌤️  Weather:")
            weather = result['weather']
            print(f"   Location: {weather['location']}")
            print(f"   Temperature: {weather['temperature']:.1f}°C")
            print(f"   Humidity: {weather['humidity']:.0f}%")
            print(f"   Wind Speed: {weather['wind_speed']:.1f} m/s")
            print(f"   Conditions: {weather['conditions']}")
            
            if 'weather_impact' in result['match_strategy']:
                wi = result['match_strategy']['weather_impact']
                print(f"\n🌊 Weather Impact:")
                for impact in wi['impacts']:
                    print(f"   • {impact}")
                print(f"   Severity: {wi['severity'].upper()}")
                print(f"   Favorable for: {wi['favorable_for'].capitalize()}")
        
        # Match Strategy
        print(f"\n🏏 Match Strategy:")
        strategy = result['match_strategy']
        print(f"   🎲 Toss Decision: {strategy['toss_decision']}")
        print(f"\n   🏏 Batting Strategy:")
        for point in strategy['batting_strategy'][:3]:
            print(f"      • {point}")
        print(f"\n   🎳 Bowling Strategy:")
        for point in strategy['bowling_strategy'][:3]:
            print(f"      • {point}")
        print(f"\n   👥 Team Composition:")
        for point in strategy['team_composition'][:3]:
            print(f"      • {point}")
        
        if strategy.get('key_factors'):
            print(f"\n   ⚠️  Key Factors:")
            for factor in strategy['key_factors']:
                print(f"      • {factor}")
    else:
        print(f"❌ Error: {response.json()}")
    
    return response.status_code == 200


def test_weather(latitude: float = 28.6139, longitude: float = 77.2090, city: str = "Mumbai"):
    """Test weather endpoint"""
    print("\n" + "="*60)
    print("🌤️  Testing Weather API")
    print("="*60)
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'city': city
    }
    
    response = requests.get(f"{BASE_URL}/api/weather", params=params)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        weather = response.json()
        print(f"\n✅ Weather Data:")
        print(f"   Location: {weather['location']}")
        print(f"   Temperature: {weather['temperature']:.1f}°C")
        print(f"   Humidity: {weather['humidity']:.0f}%")
        print(f"   Wind Speed: {weather['wind_speed']:.1f} m/s")
        print(f"   Conditions: {weather['conditions']}")
        print(f"   Rainfall: {weather['rainfall']:.1f} mm")
    elif response.status_code == 503:
        print("⚠️  Weather service unavailable (API key not configured)")
    else:
        print(f"❌ Error: {response.json()}")
    
    return response.status_code in [200, 503]


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 PITCH INSIGHT API TEST SUITE")
    print("="*60)
    
    # Test image paths
    spin_image = r"e:\pitch_insight\pitch_classification\test\spin_friendly\img000000029_jpeg.rf.5a2e387c8ed6ec3093e7282e4dfd8f5f.jpg"
    batting_image = r"e:\pitch_insight\pitch_classification\test\batting_friendly\img000000001_jpeg.rf.0dff83aee56a34c64c14ccf10f8e9d2d.jpg"
    
    results = {}
    
    # Run tests
    results['health'] = test_health_check()
    results['classes'] = test_get_classes()
    results['quick'] = test_quick_analysis(spin_image)
    results['complete'] = test_complete_analysis(batting_image, with_weather=False)
    results['complete_weather'] = test_complete_analysis(spin_image, with_weather=True)
    results['weather'] = test_weather()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
