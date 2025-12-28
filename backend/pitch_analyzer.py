"""
Pitch Feature Extraction Module
Analyzes cricket pitch images using OpenCV
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class PitchAnalyzer:
    """Extract features from cricket pitch images"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.features = {}
        
    def analyze(self, image_path: str) -> Dict:
        """
        Complete pitch analysis
        
        Args:
            image_path: Path to pitch image
            
        Returns:
            Dictionary with all extracted features
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract all features
        self.features = {
            'image_path': image_path,
            'image_shape': image.shape,
            'grass_coverage': self._detect_grass(image),
            'crack_analysis': self._detect_cracks(image),
            'moisture_level': self._analyze_moisture(image),
            'color_profile': self._analyze_color(image),
            'texture_analysis': self._analyze_texture(image),
            'brightness': self._analyze_brightness(image)
        }
        
        return self.features
    
    def _detect_grass(self, image: np.ndarray) -> Dict:
        """
        Detect grass coverage using HSV color space
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with grass metrics
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range (grass)
        # Lower bound: darker green
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # Create mask for green pixels
        mask_green = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Calculate grass coverage percentage
        total_pixels = image.shape[0] * image.shape[1]
        green_pixels = cv2.countNonZero(mask_green)
        grass_percentage = (green_pixels / total_pixels) * 100
        
        # Classify grass coverage
        if grass_percentage > 60:
            grass_level = "High"
            grass_quality = "Heavy grass coverage"
        elif grass_percentage > 30:
            grass_level = "Medium"
            grass_quality = "Moderate grass coverage"
        elif grass_percentage > 10:
            grass_level = "Low"
            grass_quality = "Sparse grass coverage"
        else:
            grass_level = "Minimal"
            grass_quality = "Bare/dry pitch"
        
        return {
            'percentage': round(grass_percentage, 2),
            'level': grass_level,
            'quality': grass_quality,
            'green_pixels': green_pixels,
            'total_pixels': total_pixels,
            'mask': mask_green
        }
    
    def _detect_cracks(self, image: np.ndarray) -> Dict:
        """
        Detect cracks using edge detection
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with crack metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to enhance cracks
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours (potential cracks)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio (cracks are usually elongated)
        crack_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio > 3:  # Elongated shapes (likely cracks)
                    crack_contours.append(contour)
        
        # Calculate crack density
        total_pixels = image.shape[0] * image.shape[1]
        crack_pixels = cv2.countNonZero(edges)
        crack_density = (crack_pixels / total_pixels) * 100
        
        # Classify crack severity
        num_cracks = len(crack_contours)
        if crack_density > 5 or num_cracks > 20:
            severity = "High"
            description = "Heavily cracked surface"
        elif crack_density > 2 or num_cracks > 10:
            severity = "Medium"
            description = "Moderate cracking"
        elif crack_density > 0.5 or num_cracks > 3:
            severity = "Low"
            description = "Minor cracks present"
        else:
            severity = "None"
            description = "No significant cracks"
        
        return {
            'density': round(crack_density, 2),
            'severity': severity,
            'description': description,
            'num_cracks': num_cracks,
            'crack_pixels': crack_pixels,
            'edges_mask': edges
        }
    
    def _analyze_moisture(self, image: np.ndarray) -> Dict:
        """
        Analyze moisture level based on color intensity
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with moisture metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Average saturation
        avg_saturation = np.mean(s)
        
        # Dark pixels indicate moisture
        dark_threshold = 100
        dark_pixels = np.sum(gray < dark_threshold)
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_percentage = (dark_pixels / total_pixels) * 100
        
        # Calculate moisture score (0-100)
        # Lower brightness + higher dark percentage = more moisture
        moisture_score = ((100 - avg_brightness/255*100) + dark_percentage) / 2
        
        # Classify moisture level
        if moisture_score > 60:
            moisture_level = "Wet"
            description = "High moisture content"
        elif moisture_score > 40:
            moisture_level = "Damp"
            description = "Moderate moisture"
        elif moisture_score > 25:
            moisture_level = "Slightly Damp"
            description = "Low moisture"
        else:
            moisture_level = "Dry"
            description = "Minimal moisture"
        
        return {
            'score': round(moisture_score, 2),
            'level': moisture_level,
            'description': description,
            'avg_brightness': round(avg_brightness, 2),
            'avg_saturation': round(avg_saturation, 2),
            'dark_pixel_percentage': round(dark_percentage, 2)
        }
    
    def _analyze_color(self, image: np.ndarray) -> Dict:
        """
        Analyze color distribution
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with color metrics
        """
        # Convert to RGB for analysis
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate mean color
        mean_color = np.mean(rgb, axis=(0, 1))
        
        # Calculate dominant color using k-means
        pixels = rgb.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Get dominant color (largest cluster)
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        dominant_color = centers[dominant_idx]
        
        # Classify pitch color
        r, g, b = mean_color
        if g > r and g > b and g > 100:
            color_type = "Green"
            description = "Grass-dominated pitch"
        elif r > 150 and g > 120 and b < 100:
            color_type = "Brown/Red"
            description = "Dry, worn pitch"
        elif r > 180 and g > 180 and b > 150:
            color_type = "Light/Pale"
            description = "Very dry pitch"
        else:
            color_type = "Mixed"
            description = "Mixed surface"
        
        return {
            'mean_rgb': mean_color.tolist(),
            'dominant_color': dominant_color.tolist(),
            'color_type': color_type,
            'description': description
        }
    
    def _analyze_texture(self, image: np.ndarray) -> Dict:
        """
        Analyze surface texture using variance
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with texture metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture using local variance
        # High variance = rough texture
        # Low variance = smooth texture
        
        # Apply Laplacian filter to detect texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Calculate local standard deviation
        mean_std = np.std(gray)
        
        # Classify texture
        if texture_variance > 1000:
            texture_type = "Very Rough"
            description = "Heavily worn/uneven surface"
        elif texture_variance > 500:
            texture_type = "Rough"
            description = "Rough, uneven surface"
        elif texture_variance > 200:
            texture_type = "Moderate"
            description = "Moderate texture"
        else:
            texture_type = "Smooth"
            description = "Smooth, even surface"
        
        return {
            'variance': round(texture_variance, 2),
            'std_dev': round(mean_std, 2),
            'type': texture_type,
            'description': description
        }
    
    def _analyze_brightness(self, image: np.ndarray) -> Dict:
        """
        Analyze overall brightness
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with brightness metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        avg_brightness = np.mean(gray)
        
        # Classify
        if avg_brightness > 180:
            level = "Very Bright"
        elif avg_brightness > 140:
            level = "Bright"
        elif avg_brightness > 100:
            level = "Moderate"
        elif avg_brightness > 60:
            level = "Dark"
        else:
            level = "Very Dark"
        
        return {
            'average': round(avg_brightness, 2),
            'level': level,
            'normalized': round(avg_brightness / 255, 2)
        }
    
    def visualize_features(self, save_path: Optional[str] = None):
        """
        Visualize all extracted features
        
        Args:
            save_path: Optional path to save visualization
        """
        if not self.features:
            print("No features to visualize. Run analyze() first.")
            return
        
        # Load original image
        image = cv2.imread(self.features['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Grass mask
        ax2 = plt.subplot(2, 3, 2)
        grass_mask = self.features['grass_coverage']['mask']
        ax2.imshow(grass_mask, cmap='Greens')
        ax2.set_title(f"Grass Coverage: {self.features['grass_coverage']['percentage']:.1f}%\n"
                     f"Level: {self.features['grass_coverage']['level']}", 
                     fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Crack detection
        ax3 = plt.subplot(2, 3, 3)
        edges = self.features['crack_analysis']['edges_mask']
        ax3.imshow(edges, cmap='gray')
        ax3.set_title(f"Cracks: {self.features['crack_analysis']['severity']}\n"
                     f"Density: {self.features['crack_analysis']['density']:.2f}%",
                     fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # Feature summary
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        summary_text = f"""
📊 PITCH FEATURE SUMMARY

🌱 Grass Coverage:
   • Percentage: {self.features['grass_coverage']['percentage']:.1f}%
   • Level: {self.features['grass_coverage']['level']}
   • {self.features['grass_coverage']['quality']}

🔺 Cracks:
   • Severity: {self.features['crack_analysis']['severity']}
   • Density: {self.features['crack_analysis']['density']:.2f}%
   • Count: {self.features['crack_analysis']['num_cracks']}
   • {self.features['crack_analysis']['description']}

💧 Moisture:
   • Level: {self.features['moisture_level']['level']}
   • Score: {self.features['moisture_level']['score']:.1f}/100
   • {self.features['moisture_level']['description']}

🎨 Color:
   • Type: {self.features['color_profile']['color_type']}
   • {self.features['color_profile']['description']}

🔲 Texture:
   • Type: {self.features['texture_analysis']['type']}
   • Variance: {self.features['texture_analysis']['variance']:.1f}
   • {self.features['texture_analysis']['description']}

💡 Brightness:
   • Level: {self.features['brightness']['level']}
   • Value: {self.features['brightness']['average']:.1f}/255
"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Color histogram
        ax5 = plt.subplot(2, 3, 5)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([rgb], [i], None, [256], [0, 256])
            ax5.plot(hist, color=color, alpha=0.7)
        ax5.set_xlabel('Pixel Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Color Distribution', fontsize=11, fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # Feature bars
        ax6 = plt.subplot(2, 3, 6)
        features_values = [
            self.features['grass_coverage']['percentage'],
            self.features['crack_analysis']['density'] * 10,  # Scale for visibility
            self.features['moisture_level']['score'],
            self.features['texture_analysis']['variance'] / 10,  # Scale
            self.features['brightness']['average'] / 2.55  # Normalize to 100
        ]
        features_names = ['Grass %', 'Cracks\n(x10)', 'Moisture', 'Texture\n(/10)', 'Brightness\n(/100)']
        colors_bar = ['#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0']
        
        bars = ax6.barh(features_names, features_values, color=colors_bar, alpha=0.7)
        ax6.set_xlabel('Value', fontsize=10)
        ax6.set_title('Feature Scores', fontsize=11, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, features_values):
            width = bar.get_width()
            ax6.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_pitch_report(self) -> str:
        """
        Generate a text report of pitch analysis
        
        Returns:
            Formatted string report
        """
        if not self.features:
            return "No analysis performed. Run analyze() first."
        
        report = f"""
{'='*60}
🏏 CRICKET PITCH ANALYSIS REPORT
{'='*60}

Image: {Path(self.features['image_path']).name}
Resolution: {self.features['image_shape'][1]}x{self.features['image_shape'][0]}

{'='*60}
📊 FEATURE ANALYSIS
{'='*60}

🌱 GRASS COVERAGE
   Coverage: {self.features['grass_coverage']['percentage']:.1f}%
   Level: {self.features['grass_coverage']['level']}
   Assessment: {self.features['grass_coverage']['quality']}

🔺 CRACK ANALYSIS
   Severity: {self.features['crack_analysis']['severity']}
   Density: {self.features['crack_analysis']['density']:.2f}%
   Number of Cracks: {self.features['crack_analysis']['num_cracks']}
   Assessment: {self.features['crack_analysis']['description']}

💧 MOISTURE LEVEL
   Level: {self.features['moisture_level']['level']}
   Score: {self.features['moisture_level']['score']:.1f}/100
   Assessment: {self.features['moisture_level']['description']}
   Brightness: {self.features['moisture_level']['avg_brightness']:.1f}/255

🎨 COLOR PROFILE
   Type: {self.features['color_profile']['color_type']}
   Assessment: {self.features['color_profile']['description']}

🔲 SURFACE TEXTURE
   Type: {self.features['texture_analysis']['type']}
   Variance: {self.features['texture_analysis']['variance']:.1f}
   Assessment: {self.features['texture_analysis']['description']}

💡 BRIGHTNESS
   Level: {self.features['brightness']['level']}
   Average: {self.features['brightness']['average']:.1f}/255

{'='*60}
"""
        return report


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pitch_analyzer.py <image_path>")
        print("Example: python pitch_analyzer.py pitch_classification/test/batting_friendly/image.jpg")
    else:
        image_path = sys.argv[1]
        
        print("🏏 Analyzing pitch image...")
        analyzer = PitchAnalyzer()
        features = analyzer.analyze(image_path)
        
        # Print report
        print(analyzer.get_pitch_report())
        
        # Visualize
        save_path = 'pitch_analysis_result.png'
        analyzer.visualize_features(save_path)
