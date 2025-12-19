"""
Complete Pitch Analysis Pipeline
Combines YOLO Detection + Feature Extraction + ML Classification with Rule-Based Adjustments
Memory-Optimized Version with Lazy Loading
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from pitch_analyzer import PitchAnalyzer
import base64
import io
import psutil
import gc


class MemoryMonitor:
    """Monitor and log memory usage"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def print_memory_usage(label="Current"):
        """Print current memory usage"""
        mem = MemoryMonitor.get_memory_usage()
        print(f"💾 {label} Memory: {mem['rss_mb']:.1f} MB (RSS), {mem['percent']:.1f}% of system")
    
    @staticmethod
    def cleanup():
        """Force garbage collection and clear CUDA cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class CompletePitchPipeline:
    """
    Complete pitch analysis pipeline:
    1. YOLO pitch detection
    2. Feature extraction (grass, cracks, moisture, etc.)
    3. ML classification with feature-based adjustments
    """
    
    def __init__(
        self,
        yolo_model_path: str = "pitch_yolov8_best.pt",
        classifier_model_path: str = "best_pitch_classifier.pth",
        device: str = None,
        lazy_load: bool = True
    ):
        """
        Initialize the pipeline with lazy loading support
        
        Args:
            yolo_model_path: Path to YOLO model
            classifier_model_path: Path to classification model
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            lazy_load: If True, models are loaded only when needed (default: True)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model_path = yolo_model_path
        self.classifier_model_path = classifier_model_path
        self.lazy_load = lazy_load
        
        # Model placeholders (not loaded yet)
        self.yolo_model = None
        self.classifier = None
        
        # Initialize feature analyzer (lightweight, no heavy models)
        self.feature_analyzer = PitchAnalyzer()
        
        # Classes
        self.classes = ['batting_friendly', 'bowling_friendly', 'seam_friendly', 'spin_friendly']
        
        # Image transform for classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"🚀 Pipeline initialized on {self.device}")
        print(f"💾 Lazy loading: {'✅ Enabled (models load on demand)' if lazy_load else '❌ Disabled'}")
        
        if not lazy_load:
            self._load_models()
        
        MemoryMonitor.print_memory_usage("Initial")
    
    def _load_models(self):
        """Load models into memory (called on-demand if lazy loading is enabled)"""
        if self.yolo_model is not None and self.classifier is not None:
            return  # Already loaded
        
        print("\n📦 Loading models into memory...")
        MemoryMonitor.print_memory_usage("Before loading")
        
        # Load YOLO model (lighter YOLOv8n variant if available)
        print("  Loading YOLO pitch detection model...")
        self.yolo_model = YOLO(self.yolo_model_path)
        
        # Load classification model - Using MobileNetV2 (lighter than ResNet18)
        print("  Loading pitch classification model (MobileNetV2)...")
        self.classifier = models.mobilenet_v2(weights=None)
        
        # Modify final layer for 4 classes
        self.classifier.classifier[1] = torch.nn.Linear(
            self.classifier.classifier[1].in_features, 4
        )
        
        # Load weights
        state_dict = torch.load(self.classifier_model_path, map_location=self.device)
        
        # Handle potential state dict format differences
        try:
            self.classifier.load_state_dict(state_dict)
        except RuntimeError:
            # If the saved model is ResNet18, we need to convert
            print("  ⚠️ Converting from ResNet18 to MobileNetV2...")
            # For now, use ResNet18 if conversion fails
            self.classifier = models.resnet18()
            self.classifier.fc = torch.nn.Linear(self.classifier.fc.in_features, 4)
            self.classifier.load_state_dict(state_dict)
        
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        # Enable model optimization
        if self.device == 'cpu':
            # Optimize for CPU inference
            self.classifier = torch.jit.optimize_for_inference(
                torch.jit.script(self.classifier)
            )
        
        MemoryMonitor.print_memory_usage("After loading")
        print("✅ Models loaded successfully!\n")
    
    def _unload_models(self):
        """Unload models from memory to free up RAM"""
        if self.yolo_model is None and self.classifier is None:
            return  # Already unloaded
        
        print("\n🧹 Unloading models from memory...")
        MemoryMonitor.print_memory_usage("Before unload")
        
        # Delete models
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None
        
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
        
        # Force cleanup
        MemoryMonitor.cleanup()
        
        MemoryMonitor.print_memory_usage("After unload")
        print("✅ Models unloaded successfully!\n")
    
    def generate_specialist_visualizations(self, pitch_region: np.ndarray, features: Dict) -> Dict:
        """
        Generate specialist analysis visualizations
        
        Args:
            pitch_region: Detected pitch region image
            features: Extracted features dictionary
            
        Returns:
            Dictionary with base64 encoded visualization images
        """
        visualizations = {}
        
        # 1. Grass Coverage Visualization
        grass_mask = features['grass_coverage']['mask']
        # Create colored overlay
        pitch_rgb = cv2.cvtColor(pitch_region, cv2.COLOR_BGR2RGB)
        grass_overlay = pitch_rgb.copy()
        # Apply green overlay where grass is detected
        green_overlay = np.zeros_like(pitch_rgb)
        green_overlay[:, :, 1] = 255  # Green channel
        grass_overlay[grass_mask > 0] = cv2.addWeighted(
            grass_overlay[grass_mask > 0], 0.6,
            green_overlay[grass_mask > 0], 0.4, 0
        )
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(grass_overlay, cv2.COLOR_RGB2BGR))
        visualizations['grass_analysis'] = base64.b64encode(buffer).decode('utf-8')
        
        # 2. Crack Detection Visualization
        edges = features['crack_analysis']['edges_mask']
        # Create heat map
        crack_visual = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
        # Blend with original
        crack_blend = cv2.addWeighted(pitch_region, 0.7, crack_visual, 0.3, 0)
        
        _, buffer = cv2.imencode('.png', crack_blend)
        visualizations['crack_analysis'] = base64.b64encode(buffer).decode('utf-8')
        
        # 3. Moisture Heat Map
        gray = cv2.cvtColor(pitch_region, cv2.COLOR_BGR2GRAY)
        # Invert so dark areas (moisture) appear hot
        moisture_map = 255 - gray
        moisture_colored = cv2.applyColorMap(moisture_map, cv2.COLORMAP_JET)
        # Blend with original
        moisture_blend = cv2.addWeighted(pitch_region, 0.6, moisture_colored, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', moisture_blend)
        visualizations['moisture_analysis'] = base64.b64encode(buffer).decode('utf-8')
        
        # 4. Original Pitch (for reference)
        _, buffer = cv2.imencode('.png', pitch_region)
        visualizations['original_pitch'] = base64.b64encode(buffer).decode('utf-8')
        
        return visualizations
    
    def detect_pitch(self, image_path: str, conf_threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Detect pitch region using YOLO
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detection
            
        Returns:
            Cropped pitch region or None if not detected
        """
        # Ensure models are loaded
        if self.yolo_model is None:
            self._load_models()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run YOLO detection
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        
        # Get best detection
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get box with highest confidence
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            
            x1, y1, x2, y2 = box
            pitch_region = image[y1:y2, x1:x2]
            
            confidence = float(boxes.conf[best_idx])
            print(f"✅ Pitch detected with {confidence:.1%} confidence")
            print(f"   Region: [{x1}, {y1}] to [{x2}, {y2}]")
            
            return pitch_region, (x1, y1, x2, y2)
        else:
            print("⚠️ No pitch detected, using full image")
            return image, None
    
    def classify_pitch(self, pitch_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Classify pitch type using ML model
        
        Args:
            pitch_image: BGR image of pitch
            
        Returns:
            (predicted_class, confidence, probabilities)
        """
        # Ensure models are loaded
        if self.classifier is None:
            self._load_models()
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(pitch_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Transform and predict
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.classifier(img_tensor)
            probabilities = F.softmax(output[0], dim=0).cpu().numpy()
        
        # Clear GPU memory after inference
        del img_tensor, output
        MemoryMonitor.cleanup()
        
        predicted_idx = probabilities.argmax()
        predicted_class = self.classes[predicted_idx]
        confidence = probabilities[predicted_idx] * 100
        
        return predicted_class, confidence, probabilities
    
    def adjust_classification_with_features(
        self,
        ml_probabilities: np.ndarray,
        features: Dict
    ) -> Tuple[str, float, np.ndarray, Dict]:
        """
        Adjust ML classification based on extracted features using cricket logic
        
        Cricket Pitch Rules:
        - High grass coverage → More swing → Bowling-friendly (helps fast bowlers)
        - Many cracks → More spin → Spin-friendly (helps spinners)
        - Dry pitch + low grass → Seam movement → Seam-friendly (helps seamers)
        - Low cracks + moderate grass → Easier batting → Batting-friendly
        
        Args:
            ml_probabilities: Original ML model probabilities
            features: Extracted pitch features
            
        Returns:
            (adjusted_class, confidence, adjusted_probs, adjustments)
        """
        # Start with ML probabilities
        adjusted_probs = ml_probabilities.copy()
        adjustments = {
            'batting_friendly': 0.0,
            'bowling_friendly': 0.0,
            'seam_friendly': 0.0,
            'spin_friendly': 0.0
        }
        reasons = []
        
        # Extract feature values
        grass_pct = features['grass_coverage']['percentage']
        grass_level = features['grass_coverage']['level']
        crack_severity = features['crack_analysis']['severity']
        crack_density = features['crack_analysis']['density']
        moisture_level = features['moisture_level']['level']
        moisture_score = features['moisture_level']['score']
        
        # RULE 1: High Grass → Bowling-friendly (swing for fast bowlers)
        if grass_pct > 60:
            adjustment = 0.15  # Strong adjustment
            adjustments['bowling_friendly'] += adjustment
            reasons.append(f"Heavy grass coverage ({grass_pct:.1f}%) favors fast bowlers (swing)")
        elif grass_pct > 40:
            adjustment = 0.08
            adjustments['bowling_friendly'] += adjustment
            reasons.append(f"Moderate grass ({grass_pct:.1f}%) helps bowlers")
        
        # RULE 2: Many Cracks → Spin-friendly
        if crack_severity in ['High', 'Medium']:
            if crack_severity == 'High':
                adjustment = 0.20  # Very strong adjustment
                adjustments['spin_friendly'] += adjustment
                reasons.append(f"Heavy cracking (severity: {crack_severity}) favors spinners")
            else:
                adjustment = 0.12
                adjustments['spin_friendly'] += adjustment
                reasons.append(f"Moderate cracking helps spin bowlers")
        
        # RULE 3: Dry Pitch + Low Grass → Seam-friendly
        if moisture_level in ['Dry', 'Slightly Damp'] and grass_pct < 30:
            adjustment = 0.15
            adjustments['seam_friendly'] += adjustment
            reasons.append(f"Dry pitch ({moisture_level}) with minimal grass favors seamers")
        
        # RULE 4: Low Cracks + Moderate Grass → Batting-friendly
        if crack_severity in ['None', 'Low'] and 20 < grass_pct < 50:
            adjustment = 0.10
            adjustments['batting_friendly'] += adjustment
            reasons.append(f"Minimal cracks with moderate grass favors batsmen")
        
        # RULE 5: Very Dry + Many Cracks → Strong Spin-friendly
        if moisture_score < 30 and crack_density > 3:
            adjustment = 0.15
            adjustments['spin_friendly'] += adjustment
            reasons.append("Dry, cracked surface ideal for spin")
        
        # RULE 6: Wet/Damp + High Grass → Strong Bowling-friendly
        if moisture_level in ['Wet', 'Damp'] and grass_pct > 50:
            adjustment = 0.12
            adjustments['bowling_friendly'] += adjustment
            reasons.append(f"Damp conditions ({moisture_level}) with grass helps swing bowlers")
        
        # Apply adjustments
        for i, cls in enumerate(self.classes):
            adjusted_probs[i] += adjustments[cls]
        
        # Normalize probabilities
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        # Get final prediction
        final_idx = adjusted_probs.argmax()
        final_class = self.classes[final_idx]
        final_confidence = adjusted_probs[final_idx] * 100
        
        adjustment_info = {
            'adjustments': adjustments,
            'reasons': reasons,
            'total_adjustment': sum(abs(v) for v in adjustments.values())
        }
        
        return final_class, final_confidence, adjusted_probs, adjustment_info
    
    def analyze(self, image_path: str, save_visualization: bool = True, auto_unload: bool = True) -> Dict:
        """
        Complete pitch analysis pipeline with memory optimization
        
        Args:
            image_path: Path to pitch image
            save_visualization: Whether to save visualization
            auto_unload: If True and lazy_load is enabled, unload models after analysis
            
        Returns:
            Complete analysis results
        """
        print("="*60)
        print("🏏 COMPLETE PITCH ANALYSIS PIPELINE")
        print("="*60)
        print(f"📸 Image: {Path(image_path).name}\n")
        
        MemoryMonitor.print_memory_usage("Start")
        
        # Step 1: Detect pitch using YOLO
        print("🔍 Step 1: Detecting pitch region with YOLO...")
        pitch_region, bbox = self.detect_pitch(image_path)
        
        # Save pitch region temporarily for feature extraction
        temp_pitch_path = "temp_pitch_region.jpg"
        cv2.imwrite(temp_pitch_path, pitch_region)
        
        # Step 2: Extract features
        print("\n📊 Step 2: Extracting pitch features...")
        features = self.feature_analyzer.analyze(temp_pitch_path)
        
        print(f"   ✅ Grass coverage: {features['grass_coverage']['percentage']:.1f}% ({features['grass_coverage']['level']})")
        print(f"   ✅ Crack severity: {features['crack_analysis']['severity']}")
        print(f"   ✅ Moisture level: {features['moisture_level']['level']}")
        print(f"   ✅ Color type: {features['color_profile']['color_type']}")
        print(f"   ✅ Surface texture: {features['texture_analysis']['type']}")
        
        # Step 3: ML Classification
        print("\n🤖 Step 3: ML model classification...")
        ml_class, ml_confidence, ml_probs = self.classify_pitch(pitch_region)
        print(f"   ML Prediction: {ml_class} ({ml_confidence:.1f}%)")
        
        # Step 4: Adjust with features
        print("\n⚙️ Step 4: Adjusting prediction with feature analysis...")
        final_class, final_confidence, final_probs, adj_info = \
            self.adjust_classification_with_features(ml_probs, features)
        
        print(f"   Final Prediction: {final_class} ({final_confidence:.1f}%)")
        
        if adj_info['reasons']:
            print(f"\n   📋 Adjustment Reasons:")
            for reason in adj_info['reasons']:
                print(f"      • {reason}")
        
        # Generate specialist visualizations
        print("\n🔬 Step 5: Generating specialist visualizations...")
        specialist_viz = self.generate_specialist_visualizations(pitch_region, features)
        
        # Compile results
        results = {
            'image_path': image_path,
            'pitch_detection': {
                'detected': bbox is not None,
                'bbox': bbox,
                'pitch_region': pitch_region
            },
            'features': features,
            'ml_classification': {
                'prediction': ml_class,
                'confidence': ml_confidence,
                'probabilities': ml_probs
            },
            'final_classification': {
                'prediction': final_class,
                'confidence': final_confidence,
                'probabilities': final_probs,
                'adjustment_info': adj_info
            },
            'specialist_analysis': specialist_viz
        }
        
        # Visualization
        if save_visualization:
            print("\n📊 Generating visualization...")
            self.visualize_complete_analysis(image_path, results)
        
        # Unload models if lazy loading is enabled and auto_unload is True
        if self.lazy_load and auto_unload:
            self._unload_models()
        
        MemoryMonitor.print_memory_usage("End")
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        
        return results
    
    def visualize_complete_analysis(self, image_path: str, results: Dict):
        """
        Create comprehensive visualization of all analysis steps
        """
        # Load original image
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        pitch_region = results['pitch_detection']['pitch_region']
        pitch_rgb = cv2.cvtColor(pitch_region, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Original image with bbox
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(original_rgb)
        if results['pitch_detection']['bbox']:
            x1, y1, x2, y2 = results['pitch_detection']['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.set_title('Original Image\n(Pitch Detected)', fontsize=10, fontweight='bold')
        else:
            ax1.set_title('Original Image', fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # 2. Detected pitch region
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(pitch_rgb)
        ax2.set_title('Detected Pitch Region', fontsize=10, fontweight='bold')
        ax2.axis('off')
        
        # 3. Grass mask
        ax3 = plt.subplot(3, 4, 3)
        grass_mask = results['features']['grass_coverage']['mask']
        ax3.imshow(grass_mask, cmap='Greens')
        ax3.set_title(f"Grass: {results['features']['grass_coverage']['percentage']:.1f}%\n"
                     f"{results['features']['grass_coverage']['level']}", 
                     fontsize=10, fontweight='bold')
        ax3.axis('off')
        
        # 4. Crack detection
        ax4 = plt.subplot(3, 4, 4)
        edges = results['features']['crack_analysis']['edges_mask']
        ax4.imshow(edges, cmap='hot')
        ax4.set_title(f"Cracks: {results['features']['crack_analysis']['severity']}\n"
                     f"Density: {results['features']['crack_analysis']['density']:.1f}%",
                     fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        # 5. Feature Summary
        ax5 = plt.subplot(3, 4, 5)
        ax5.axis('off')
        feature_text = f"""
FEATURE ANALYSIS

Grass Coverage: {results['features']['grass_coverage']['percentage']:.1f}%
Level: {results['features']['grass_coverage']['level']}

Crack Severity: {results['features']['crack_analysis']['severity']}
Crack Density: {results['features']['crack_analysis']['density']:.1f}%

Moisture: {results['features']['moisture_level']['level']}
Score: {results['features']['moisture_level']['score']:.1f}/100

Color: {results['features']['color_profile']['color_type']}
Texture: {results['features']['texture_analysis']['type']}
Brightness: {results['features']['brightness']['level']}
"""
        ax5.text(0.1, 0.5, feature_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 6. ML Classification
        ax6 = plt.subplot(3, 4, 6)
        ml_probs = results['ml_classification']['probabilities'] * 100
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        bars = ax6.barh(self.classes, ml_probs, color=colors, alpha=0.7)
        ax6.set_xlabel('Probability (%)', fontsize=9)
        ax6.set_title(f"ML Prediction\n{results['ml_classification']['prediction']}", 
                     fontsize=10, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
        for bar, prob in zip(bars, ml_probs):
            ax6.text(prob + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=8)
        
        # 7. Feature Adjustments
        ax7 = plt.subplot(3, 4, 7)
        adjustments = results['final_classification']['adjustment_info']['adjustments']
        adj_values = [adjustments[cls]*100 for cls in self.classes]
        colors_adj = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in adj_values]
        bars = ax7.barh(self.classes, adj_values, color=colors_adj, alpha=0.7)
        ax7.set_xlabel('Adjustment (%)', fontsize=9)
        ax7.set_title('Feature-Based Adjustments', fontsize=10, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3)
        ax7.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        for bar, adj in zip(bars, adj_values):
            if abs(adj) > 0.5:
                ax7.text(adj + (0.5 if adj > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                        f'{adj:+.1f}%', va='center', fontsize=8)
        
        # 8. Final Classification
        ax8 = plt.subplot(3, 4, 8)
        final_probs = results['final_classification']['probabilities'] * 100
        bars = ax8.barh(self.classes, final_probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax8.set_xlabel('Probability (%)', fontsize=9)
        ax8.set_title(f"FINAL PREDICTION\n{results['final_classification']['prediction']}", 
                     fontsize=10, fontweight='bold', color='darkgreen')
        ax8.grid(axis='x', alpha=0.3)
        for bar, prob in zip(bars, final_probs):
            ax8.text(prob + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=8, fontweight='bold')
        
        # 9. Adjustment Reasons
        ax9 = plt.subplot(3, 4, (9, 10))
        ax9.axis('off')
        reasons = results['final_classification']['adjustment_info']['reasons']
        if reasons:
            reasons_text = "ADJUSTMENT REASONS:\n\n" + "\n\n".join(f"• {r}" for r in reasons)
        else:
            reasons_text = "No feature-based adjustments applied.\nML prediction stands."
        ax9.text(0.1, 0.5, reasons_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 10. Comparison
        ax10 = plt.subplot(3, 4, (11, 12))
        comparison_data = {
            'ML Only': results['ml_classification']['confidence'],
            'ML + Features': results['final_classification']['confidence']
        }
        bars = ax10.bar(comparison_data.keys(), comparison_data.values(), 
                       color=['#2196F3', '#4CAF50'], alpha=0.7)
        ax10.set_ylabel('Confidence (%)', fontsize=10)
        ax10.set_title('Confidence Comparison', fontsize=10, fontweight='bold')
        ax10.set_ylim(0, 100)
        ax10.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('🏏 COMPLETE PITCH ANALYSIS PIPELINE', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        save_path = 'complete_pitch_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved to: {save_path}")
        plt.show()
    
    def generate_comprehensive_report(self, image_path: str, results: Dict, weather_data: Dict = None) -> str:
        """
        Generate comprehensive visual report with all analysis components including weather
        
        Args:
            image_path: Path to original image
            results: Complete analysis results
            weather_data: Optional weather data dictionary
            
        Returns:
            Path to saved report image
        """
        # Load original image
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        pitch_region = results['pitch_detection']['pitch_region']
        pitch_rgb = cv2.cvtColor(pitch_region, cv2.COLOR_BGR2RGB)
        
        # Create figure with better layout
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('white')
        
        # Title
        fig.suptitle('🏏 COMPLETE PITCH ANALYSIS PIPELINE', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Row 1: Original and Detected Pitch with Large Images
        ax1 = plt.subplot(4, 5, 1)
        ax1.imshow(original_rgb)
        if results['pitch_detection']['bbox']:
            x1, y1, x2, y2 = results['pitch_detection']['bbox']
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='red', linewidth=3)
            ax1.add_patch(rect)
        ax1.set_title('Original Image\n(Pitch Detected)', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(4, 5, 2)
        ax2.imshow(pitch_rgb)
        ax2.set_title('Detected Pitch Region', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Row 1: Feature Analysis Images
        ax3 = plt.subplot(4, 5, 3)
        grass_mask = results['features']['grass_coverage']['mask']
        ax3.imshow(grass_mask, cmap='Greens')
        ax3.set_title(f"Grass: {results['features']['grass_coverage']['percentage']:.1f}%\n{results['features']['grass_coverage']['level']}", 
                     fontsize=10, fontweight='bold', color='darkgreen')
        ax3.axis('off')
        
        ax4 = plt.subplot(4, 5, 4)
        edges = results['features']['crack_analysis']['edges_mask']
        ax4.imshow(edges, cmap='hot')
        ax4.set_title(f"Cracks: {results['features']['crack_analysis']['severity']}\nDensity: {results['features']['crack_analysis']['density']:.1f}%",
                     fontsize=10, fontweight='bold', color='#d32f2f')
        ax4.axis('off')
        
        ax5 = plt.subplot(4, 5, 5)
        # Create moisture visualization
        gray = cv2.cvtColor(pitch_region, cv2.COLOR_BGR2GRAY)
        moisture_map = 255 - gray
        ax5.imshow(moisture_map, cmap='Blues')
        ax5.set_title(f"Moisture: {results['features']['moisture_level']['level']}\nScore: {results['features']['moisture_level']['score']:.1f}/100",
                     fontsize=10, fontweight='bold', color='#1976d2')
        ax5.axis('off')
        
        # Row 2: Feature Statistics
        ax6 = plt.subplot(4, 5, (6, 7))
        ax6.axis('off')
        feature_text = f"""FEATURE ANALYSIS STATISTICS

🌱 Grass Coverage:
   • Percentage: {results['features']['grass_coverage']['percentage']:.1f}%
   • Level: {results['features']['grass_coverage']['level']}
   • Quality: {results['features']['grass_coverage']['quality']}

⚡ Crack Analysis:
   • Severity: {results['features']['crack_analysis']['severity']}
   • Density: {results['features']['crack_analysis']['density']:.2f}%
   • Count: {results['features']['crack_analysis']['num_cracks']}

💧 Moisture Level:
   • Level: {results['features']['moisture_level']['level']}
   • Score: {results['features']['moisture_level']['score']:.1f}/100
   • Description: {results['features']['moisture_level']['description']}

🎨 Color Profile:
   • Type: {results['features']['color_profile']['color_type']}
   • Description: {results['features']['color_profile']['description']}

🔲 Texture Analysis:
   • Type: {results['features']['texture_analysis']['type']}
   • Variance: {results['features']['texture_analysis']['variance']:.2f}

☀️ Brightness:
   • Level: {results['features']['brightness']['level']}
   • Average: {results['features']['brightness']['average']:.2f}
"""
        ax6.text(0.05, 0.5, feature_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8, pad=1))
        
        # Weather Data (if available)
        if weather_data:
            ax7 = plt.subplot(4, 5, (8, 9))
            ax7.axis('off')
            weather_text = f"""WEATHER CONDITIONS

📍 Location: {weather_data.get('location', 'N/A')}

🌡️ Temperature: {weather_data.get('temperature', 'N/A')}°C
💧 Humidity: {weather_data.get('humidity', 'N/A')}%
🌧️ Rainfall: {weather_data.get('rainfall', 0):.1f} mm
💨 Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s
☁️ Conditions: {weather_data.get('conditions', 'N/A')}

Impact on Pitch:
• {"High moisture aids seam movement" if weather_data.get('humidity', 0) > 70 else "Dry conditions favor spin"}
• {"Overcast helps swing bowling" if 'cloud' in str(weather_data.get('conditions', '')).lower() else "Clear skies aid batting"}
"""
            ax7.text(0.05, 0.5, weather_text, transform=ax7.transAxes,
                    fontsize=9, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8, pad=1))
        
        # Row 3: ML Prediction
        ax8 = plt.subplot(4, 5, 11)
        ml_probs = results['ml_classification']['probabilities'] * 100
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        bars = ax8.barh(self.classes, ml_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax8.set_xlabel('Probability (%)', fontsize=10, fontweight='bold')
        ax8.set_title(f"ML Prediction\n{results['ml_classification']['prediction'].replace('_', ' ').title()}", 
                     fontsize=11, fontweight='bold', color='#1976d2')
        ax8.grid(axis='x', alpha=0.3, linestyle='--')
        ax8.set_xlim(0, 100)
        for bar, prob in zip(bars, ml_probs):
            ax8.text(prob + 2, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # Feature Adjustments
        ax9 = plt.subplot(4, 5, 12)
        adjustments = results['final_classification']['adjustment_info']['adjustments']
        adj_values = [adjustments[cls]*100 for cls in self.classes]
        colors_adj = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in adj_values]
        bars = ax9.barh(self.classes, adj_values, color=colors_adj, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax9.set_xlabel('Adjustment (%)', fontsize=10, fontweight='bold')
        ax9.set_title('Feature-Based Adjustments', fontsize=11, fontweight='bold')
        ax9.grid(axis='x', alpha=0.3, linestyle='--')
        ax9.axvline(x=0, color='black', linestyle='-', linewidth=2)
        for bar, adj in zip(bars, adj_values):
            if abs(adj) > 0.5:
                ax9.text(adj + (0.5 if adj > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                        f'{adj:+.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # Final Classification
        ax10 = plt.subplot(4, 5, 13)
        final_probs = results['final_classification']['probabilities'] * 100
        bars = ax10.barh(self.classes, final_probs, color=colors, alpha=0.9, edgecolor='darkgreen', linewidth=2.5)
        ax10.set_xlabel('Probability (%)', fontsize=10, fontweight='bold')
        ax10.set_title(f"FINAL PREDICTION\n{results['final_classification']['prediction'].replace('_', ' ').upper()}", 
                     fontsize=11, fontweight='bold', color='darkgreen')
        ax10.grid(axis='x', alpha=0.3, linestyle='--')
        ax10.set_xlim(0, 100)
        for bar, prob in zip(bars, final_probs):
            ax10.text(prob + 2, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # Confidence Comparison
        ax11 = plt.subplot(4, 5, (14, 15))
        comparison_data = {
            'ML Only': results['ml_classification']['confidence'],
            'ML + Features': results['final_classification']['confidence']
        }
        bars = ax11.bar(comparison_data.keys(), comparison_data.values(), 
                       color=['#64B5F6', '#66BB6A'], alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
        ax11.set_ylabel('Confidence (%)', fontsize=11, fontweight='bold')
        ax11.set_title('Confidence Comparison', fontsize=11, fontweight='bold')
        ax11.set_ylim(0, 100)
        ax11.grid(axis='y', alpha=0.3, linestyle='--')
        for bar in bars:
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{height:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        # Row 4: Adjustment Reasons
        ax12 = plt.subplot(4, 5, (16, 20))
        ax12.axis('off')
        reasons = results['final_classification']['adjustment_info']['reasons']
        if reasons:
            reasons_text = "ADJUSTMENT REASONS:\n\n" + "\n\n".join(f"• {r}" for r in reasons)
        else:
            reasons_text = "✓ No feature-based adjustments applied.\n✓ ML prediction confidence is high.\n✓ Features align with ML classification."
        ax12.text(0.05, 0.5, reasons_text, transform=ax12.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.8, pad=1.5))
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        
        # Save with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'pitch_analysis_report_{timestamp}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"   ✅ Comprehensive report saved to: {save_path}")
        
        return save_path
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate comprehensive text report
        """
        report = f"""
{'='*70}
🏏 COMPLETE CRICKET PITCH ANALYSIS REPORT
{'='*70}

Image: {Path(results['image_path']).name}

{'='*70}
🔍 STEP 1: PITCH DETECTION
{'='*70}
Status: {'✅ Detected' if results['pitch_detection']['detected'] else '⚠️ Using full image'}
{'Bounding Box: ' + str(results['pitch_detection']['bbox']) if results['pitch_detection']['bbox'] else ''}

{'='*70}
📊 STEP 2: FEATURE EXTRACTION
{'='*70}

🌱 Grass Coverage:
   • Percentage: {results['features']['grass_coverage']['percentage']:.1f}%
   • Level: {results['features']['grass_coverage']['level']}
   • Assessment: {results['features']['grass_coverage']['quality']}

🔺 Crack Analysis:
   • Severity: {results['features']['crack_analysis']['severity']}
   • Density: {results['features']['crack_analysis']['density']:.2f}%
   • Count: {results['features']['crack_analysis']['num_cracks']}
   • Assessment: {results['features']['crack_analysis']['description']}

💧 Moisture Level:
   • Level: {results['features']['moisture_level']['level']}
   • Score: {results['features']['moisture_level']['score']:.1f}/100
   • Assessment: {results['features']['moisture_level']['description']}

🎨 Color Profile:
   • Type: {results['features']['color_profile']['color_type']}
   • Assessment: {results['features']['color_profile']['description']}

🔲 Surface Texture:
   • Type: {results['features']['texture_analysis']['type']}
   • Variance: {results['features']['texture_analysis']['variance']:.1f}

💡 Brightness:
   • Level: {results['features']['brightness']['level']}
   • Value: {results['features']['brightness']['average']:.1f}/255

{'='*70}
🤖 STEP 3: ML CLASSIFICATION
{'='*70}
Prediction: {results['ml_classification']['prediction'].upper()}
Confidence: {results['ml_classification']['confidence']:.1f}%

Probabilities:
"""
        for cls, prob in zip(self.classes, results['ml_classification']['probabilities']):
            report += f"   • {cls:20s}: {prob*100:5.1f}%\n"
        
        report += f"""
{'='*70}
⚙️ STEP 4: FEATURE-BASED ADJUSTMENT
{'='*70}
"""
        if results['final_classification']['adjustment_info']['reasons']:
            report += "Applied Adjustments:\n"
            for reason in results['final_classification']['adjustment_info']['reasons']:
                report += f"   • {reason}\n"
        else:
            report += "No adjustments applied.\n"
        
        report += f"""
{'='*70}
🎯 FINAL PREDICTION
{'='*70}
PITCH TYPE: {results['final_classification']['prediction'].upper().replace('_', ' ')}
CONFIDENCE: {results['final_classification']['confidence']:.1f}%

Final Probabilities:
"""
        for cls, prob in zip(self.classes, results['final_classification']['probabilities']):
            report += f"   • {cls:20s}: {prob*100:5.1f}%\n"
        
        report += f"""
{'='*70}
📝 MATCH IMPLICATIONS
{'='*70}
"""
        
        final_class = results['final_classification']['prediction']
        
        if final_class == 'batting_friendly':
            report += """
✅ BATTING-FRIENDLY PITCH

Match Strategy:
   • Batsmen can play freely with confidence
   • Good for stroke play and run scoring
   • Bowlers need to be patient and disciplined
   • Consider batting first if winning the toss

Team Selection:
   • Include aggressive batsmen
   • Select quality fast bowlers who can extract bounce
   • Spinners may have limited impact
"""
        elif final_class == 'bowling_friendly':
            report += """
✅ BOWLING-FRIENDLY PITCH

Match Strategy:
   • Bowlers, especially fast bowlers, will dominate
   • Expect swing and seam movement
   • Batting will be challenging
   • Consider bowling first if winning the toss

Team Selection:
   • Include quality pace attack
   • Select technically sound batsmen
   • All-rounders valuable for balance
"""
        elif final_class == 'spin_friendly':
            report += """
✅ SPIN-FRIENDLY PITCH

Match Strategy:
   • Spinners will be key wicket-takers
   • Pitch will deteriorate and assist spin more on Day 4-5
   • Batting becomes harder as match progresses
   • Toss is crucial - bat first advantage

Team Selection:
   • Include 2-3 quality spinners
   • Select batsmen good against spin
   • Consider extra spinner over pace bowler
"""
        else:  # seam_friendly
            report += """
✅ SEAM-FRIENDLY PITCH

Match Strategy:
   • Seam bowlers will get good movement
   • Lateral movement off the pitch expected
   • Requires disciplined batting technique
   • First session crucial for setting tone

Team Selection:
   • Include seam bowlers with good control
   • Select batsmen with solid technique
   • Opening bowlers will be crucial
"""
        
        report += f"\n{'='*70}\n"
        
        return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python complete_pipeline.py <image_path>")
        print("Example: python complete_pipeline.py pitch_classification/test/batting_friendly/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize pipeline
    pipeline = CompletePitchPipeline(
        yolo_model_path="pitch_yolov8_best.pt",
        classifier_model_path="best_pitch_classifier.pth"
    )
    
    # Run complete analysis
    results = pipeline.analyze(image_path, save_visualization=True)
    
    # Generate and print report
    report = pipeline.generate_report(results)
    print(report)
    
    # Save report
    report_path = 'pitch_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Report saved to: {report_path}")
