"""
Feature Extraction Validation Script
Tests all feature extraction methods to ensure they work properly
"""

import cv2
import numpy as np
from pitch_analyzer import PitchAnalyzer
import sys

def create_test_image(width=640, height=480):
    """Create a synthetic test image with known characteristics"""
    # Create a base brown pitch
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array([120, 140, 160], dtype=np.uint8)
    
    # Ensure contiguous array for OpenCV
    image = np.ascontiguousarray(image)
    
    # Add some green grass (top 30%)
    grass_height = int(height * 0.3)
    image[:grass_height, :] = [40, 120, 60]  # Green color
    
    # Add some cracks (dark lines)
    for i in range(5):
        y = int(height * 0.4) + i * 50
        cv2.line(image, (0, y), (width, y + 20), (80, 80, 80), 2)
    
    # Add some darker areas (moisture simulation)
    cv2.rectangle(image, (100, 200), (300, 350), (60, 70, 80), -1)
    
    return image

def validate_feature_output(feature_name, result, required_keys):
    """Validate that a feature extraction result has all required keys"""
    print(f"\n{'='*60}")
    print(f"Testing: {feature_name}")
    print(f"{'='*60}")
    
    issues = []
    
    # Check if result is a dictionary
    if not isinstance(result, dict):
        issues.append(f"❌ ERROR: Result is not a dictionary, got {type(result)}")
        return issues
    
    # Check required keys
    for key in required_keys:
        if key not in result:
            issues.append(f"❌ Missing key: '{key}'")
        else:
            value = result[key]
            print(f"  ✓ {key}: {value} (type: {type(value).__name__})")
            
            # Additional validation
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value):
                    issues.append(f"⚠️  WARNING: '{key}' is NaN")
                elif np.isinf(value):
                    issues.append(f"⚠️  WARNING: '{key}' is infinite")
            elif value is None:
                issues.append(f"⚠️  WARNING: '{key}' is None")
    
    if not issues:
        print(f"✅ {feature_name}: PASSED")
    else:
        print(f"❌ {feature_name}: FAILED")
        for issue in issues:
            print(f"  {issue}")
    
    return issues

def test_all_features():
    """Test all feature extraction methods"""
    print("\n" + "="*70)
    print("🧪 FEATURE EXTRACTION VALIDATION TEST")
    print("="*70)
    
    all_issues = []
    
    # Create test image
    print("\n📸 Creating synthetic test image...")
    test_image = create_test_image()
    test_path = "test_pitch_synthetic.jpg"
    cv2.imwrite(test_path, test_image)
    print(f"   ✓ Test image saved: {test_path}")
    print(f"   ✓ Image shape: {test_image.shape}")
    
    # Initialize analyzer
    print("\n🔧 Initializing PitchAnalyzer...")
    analyzer = PitchAnalyzer()
    print("   ✓ Analyzer initialized")
    
    # Test individual feature extraction methods
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL FEATURE METHODS")
    print("="*70)
    
    # 1. Test Grass Detection
    grass_result = analyzer._detect_grass(test_image)
    issues = validate_feature_output(
        "Grass Detection",
        grass_result,
        ['percentage', 'level', 'quality', 'green_pixels', 'total_pixels', 'mask']
    )
    all_issues.extend(issues)
    
    # Validate grass percentage range
    if 'percentage' in grass_result:
        if not (0 <= grass_result['percentage'] <= 100):
            all_issues.append("❌ Grass percentage out of range [0, 100]")
        else:
            print(f"  ✓ Grass percentage in valid range: {grass_result['percentage']:.2f}%")
    
    # 2. Test Crack Detection
    crack_result = analyzer._detect_cracks(test_image)
    issues = validate_feature_output(
        "Crack Detection",
        crack_result,
        ['density', 'severity', 'description', 'num_cracks', 'num_major_cracks', 'num_thin_cracks', 'crack_pixels', 'edges_mask']
    )
    all_issues.extend(issues)
    
    # Validate crack density range
    if 'density' in crack_result:
        if not (0 <= crack_result['density'] <= 100):
            all_issues.append("❌ Crack density out of range [0, 100]")
        else:
            print(f"  ✓ Crack density in valid range: {crack_result['density']:.2f}%")
    
    # 3. Test Moisture Analysis
    moisture_result = analyzer._analyze_moisture(test_image)
    issues = validate_feature_output(
        "Moisture Analysis",
        moisture_result,
        ['score', 'level', 'description', 'avg_brightness', 'avg_saturation', 'dark_pixel_percentage']
    )
    all_issues.extend(issues)
    
    # Validate moisture score range
    if 'score' in moisture_result:
        if not (0 <= moisture_result['score'] <= 100):
            all_issues.append("❌ Moisture score out of range [0, 100]")
        else:
            print(f"  ✓ Moisture score in valid range: {moisture_result['score']:.2f}")
    
    # 4. Test Color Analysis
    color_result = analyzer._analyze_color(test_image)
    issues = validate_feature_output(
        "Color Analysis",
        color_result,
        ['mean_rgb', 'dominant_color', 'color_type', 'description']
    )
    all_issues.extend(issues)
    
    # 5. Test Texture Analysis
    texture_result = analyzer._analyze_texture(test_image)
    issues = validate_feature_output(
        "Texture Analysis",
        texture_result,
        ['variance', 'std_dev', 'type', 'description']
    )
    all_issues.extend(issues)
    
    # 6. Test Brightness Analysis
    brightness_result = analyzer._analyze_brightness(test_image)
    issues = validate_feature_output(
        "Brightness Analysis",
        brightness_result,
        ['average', 'level', 'normalized']
    )
    all_issues.extend(issues)
    
    # Validate brightness range
    if 'average' in brightness_result:
        if not (0 <= brightness_result['average'] <= 255):
            all_issues.append("❌ Brightness average out of range [0, 255]")
        else:
            print(f"  ✓ Brightness in valid range: {brightness_result['average']:.2f}")
    
    # Test complete analysis
    print("\n" + "="*70)
    print("TESTING COMPLETE ANALYSIS PIPELINE")
    print("="*70)
    
    complete_result = analyzer.analyze(test_path)
    issues = validate_feature_output(
        "Complete Analysis",
        complete_result,
        ['image_path', 'image_shape', 'grass_coverage', 'crack_analysis', 'moisture_level', 'color_profile', 'texture_analysis', 'brightness']
    )
    all_issues.extend(issues)
    
    # Final Report
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    if not all_issues:
        print("\n✅ ALL TESTS PASSED!")
        print("   All feature extraction methods are working correctly.")
        print(f"   Total features tested: 6")
        return True
    else:
        print(f"\n❌ TESTS FAILED!")
        print(f"   Total issues found: {len(all_issues)}")
        print("\nIssues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        return False

def test_with_real_image(image_path):
    """Test with a real pitch image"""
    print("\n" + "="*70)
    print(f"🏏 TESTING WITH REAL IMAGE: {image_path}")
    print("="*70)
    
    try:
        analyzer = PitchAnalyzer()
        result = analyzer.analyze(image_path)
        
        print("\n✅ Analysis completed successfully!")
        print("\nResults Summary:")
        print(f"  • Grass Coverage: {result['grass_coverage']['percentage']:.1f}% ({result['grass_coverage']['level']})")
        print(f"  • Cracks: {result['crack_analysis']['num_cracks']} ({result['crack_analysis']['severity']})")
        print(f"  • Crack Density: {result['crack_analysis']['density']:.2f}%")
        print(f"  • Moisture: {result['moisture_level']['score']:.1f}/100 ({result['moisture_level']['level']})")
        print(f"  • Color: {result['color_profile']['color_type']}")
        print(f"  • Texture: {result['texture_analysis']['type']}")
        print(f"  • Brightness: {result['brightness']['average']:.1f}/255 ({result['brightness']['level']})")
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run synthetic test
    success = test_all_features()
    
    # If test passed and real image provided, test with real image
    if success and len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_with_real_image(image_path)
    elif len(sys.argv) > 1:
        print("\n⚠️  Real image test skipped due to synthetic test failures")
    
    # Clean up
    import os
    if os.path.exists("test_pitch_synthetic.jpg"):
        os.remove("test_pitch_synthetic.jpg")
        print("\n🧹 Cleaned up test files")
    
    sys.exit(0 if success else 1)
