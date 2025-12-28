"""
Complete Pitch Analysis Pipeline - ONNX Optimized Version
Combines YOLO Detection + Feature Extraction + ML Classification with Rule-Based Adjustments
Uses ONNX Runtime for faster loading and inference
Optimized for deployment on limited memory environments (512MB RAM)
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional
from pitch_analyzer import PitchAnalyzer


class CompletePitchPipeline:
    """
    Complete pitch analysis pipeline using ONNX:
    1. YOLO pitch detection (ONNX)
    2. Feature extraction (grass, cracks, moisture, etc.)
    3. ML classification (ONNX) with feature-based adjustments
    """
    
    def __init__(
        self,
        yolo_model_path: str = "pitch_yolov8_best.onnx",
        classifier_model_path: str = "pitch_classifier.onnx",
        use_gpu: bool = False
    ):
        """
        Initialize the pipeline with ONNX models
        
        Args:
            yolo_model_path: Path to YOLO ONNX model
            classifier_model_path: Path to classification ONNX model
            use_gpu: Whether to use GPU (CUDA) for inference
        """
        print(f"🚀 Initializing ONNX pipeline...")
        
        # Setup ONNX Runtime providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Load YOLO model
        print("📦 Loading YOLO ONNX model...")
        self.yolo_session = ort.InferenceSession(yolo_model_path, providers=providers)
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        self.yolo_input_shape = self.yolo_session.get_inputs()[0].shape
        
        # Load classification model
        print("📦 Loading pitch classifier ONNX model...")
        self.classifier_session = ort.InferenceSession(classifier_model_path, providers=providers)
        self.classifier_input_name = self.classifier_session.get_inputs()[0].name
        
        # Initialize feature analyzer
        self.feature_analyzer = PitchAnalyzer()
        
        # Classes
        self.classes = ['batting_friendly', 'bowling_friendly', 'seam_friendly', 'spin_friendly']
        
        # Normalization parameters (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        
        print("✅ ONNX Pipeline initialized successfully!\n")
    
    def preprocess_yolo_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO ONNX model
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed image tensor, scale factor, and padding offsets (left, top)
        """
        # Get target size from model input shape
        target_h, target_w = 640, 640  # YOLOv8 default
        
        # Calculate scale
        h, w = image.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_h, pad_w = target_h - new_h, target_w - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to RGB and normalize
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        image_transposed = np.transpose(image_norm, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0).astype(np.float32)
        
        return image_batch, scale, (left, top)
    
    def postprocess_yolo_output(self, output: np.ndarray, original_shape: Tuple[int, int], 
                                 scale: float, padding: Tuple[int, int], conf_threshold: float = 0.5) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Postprocess YOLO ONNX output to get bounding boxes
        
        Args:
            output: Raw YOLO output
            original_shape: Original image shape (h, w)
            scale: Scale factor used in preprocessing
            padding: Padding offsets (left, top)
            conf_threshold: Confidence threshold
            
        Returns:
            None or (bbox, confidence)
        """
        # YOLOv8 output format: [1, 84, 8400] or [1, 8400, 84]
        # 84 = 4 (bbox) + 80 (classes) but we only have 1 class
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Ensure shape is [num_detections, features]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # For single-class YOLO: shape should be [8400, 5] where 5 = [x, y, w, h, conf]
        # Extract boxes and confidence
        if output.shape[1] >= 5:
            boxes = output[:, :4]  # x_center, y_center, width, height (in 640x640 scale)
            confidences = output[:, 4]  # confidence scores
        else:
            # Fallback: assume all features are boxes with max score as confidence
            boxes = output[:, :4]
            confidences = output[:, 4:].max(axis=1) if output.shape[1] > 4 else np.ones(len(boxes))
        
        # Filter by confidence threshold
        mask = confidences >= conf_threshold
        if not mask.any():
            print(f"   No detections above threshold {conf_threshold}")
            return None
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        
        # Get best detection
        best_idx = confidences.argmax()
        box = boxes[best_idx]
        confidence = confidences[best_idx]
        
        # YOLOv8 boxes are in the format: [x_center, y_center, width, height] 
        # in the 640x640 input image coordinate system (WITH padding)
        x_center, y_center, width, height = box
        
        # CRITICAL: Remove letterbox padding offsets BEFORE scaling
        left_pad, top_pad = padding
        x_center_no_pad = x_center - left_pad
        y_center_no_pad = y_center - top_pad
        
        # Convert to original image coordinates
        x_center_orig = x_center_no_pad / scale
        y_center_orig = y_center_no_pad / scale
        width_orig = width / scale
        height_orig = height / scale
        
        # Convert from center format to corner format
        x1 = int(x_center_orig - width_orig / 2)
        y1 = int(y_center_orig - height_orig / 2)
        x2 = int(x_center_orig + width_orig / 2)
        y2 = int(y_center_orig + height_orig / 2)
        
        # Clip to image bounds
        h, w = original_shape
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        print(f"   Debug: Raw box (with padding): {box}, Confidence: {confidence:.4f}")
        print(f"   Debug: Final bbox (original image): ({x1}, {y1}, {x2}, {y2})")
        
        return (x1, y1, x2, y2), float(confidence)
    
    def detect_pitch(self, image_path: str, conf_threshold: float = 0.25) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect pitch region using YOLO ONNX model
        Always returns a cropped region - either detected pitch or center crop
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detection (default: 0.25)
            
        Returns:
            (cropped_pitch_region, bbox) - always returns a crop, never full image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        input_tensor, scale, padding = self.preprocess_yolo_image(image)
        
        # Run inference
        outputs = self.yolo_session.run(None, {self.yolo_input_name: input_tensor})
        
        # Postprocess
        result = self.postprocess_yolo_output(outputs[0], image.shape[:2], scale, padding, conf_threshold)
        
        if result:
            bbox, confidence = result
            x1, y1, x2, y2 = bbox
            
            # Validate bbox has reasonable size
            width = x2 - x1
            height = y2 - y1
            if width < 10 or height < 10:
                print(f"⚠️ Detected region too small ({width}x{height}), using full image as fallback")
                return image, None
            
            pitch_region = image[y1:y2, x1:x2]
            
            print(f"✅ Pitch detected with {confidence*100:.1f}% confidence")
            print(f"   Cropped region: [{x1}, {y1}] to [{x2}, {y2}] (size: {width}x{height})")
            
            return pitch_region, (x1, y1, x2, y2)
        else:
            print("⚠️ No pitch detected with sufficient confidence")
            print(f"   Using center crop as fallback")
            # Use center crop of image as fallback
            h, w = image.shape[:2]
            crop_size = min(h, w) * 0.8  # 80% of smaller dimension
            x1 = int((w - crop_size) / 2)
            y1 = int((h - crop_size) / 2)
            x2 = int(x1 + crop_size)
            y2 = int(y1 + crop_size)
            pitch_region = image[y1:y2, x1:x2]
            print(f"   Center crop region: [{x1}, {y1}] to [{x2}, {y2}]")
            return pitch_region, (x1, y1, x2, y2)
    
    def preprocess_classifier_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for classifier ONNX model
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Convert to float and normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        image_chw = np.transpose(image_float, (2, 0, 1))
        
        # Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0)
        
        # Normalize with ImageNet mean/std
        image_normalized = (image_batch - self.mean) / self.std
        
        return image_normalized.astype(np.float32)
    
    def classify_pitch(self, pitch_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Classify pitch type using ONNX model
        
        Args:
            pitch_image: BGR image of pitch
            
        Returns:
            (predicted_class, confidence, probabilities)
        """
        # Preprocess
        input_tensor = self.preprocess_classifier_image(pitch_image)
        
        # Run inference
        outputs = self.classifier_session.run(None, {self.classifier_input_name: input_tensor})
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
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
            ml_probabilities: Original ML probabilities
            features: Extracted pitch features
            
        Returns:
            (final_class, confidence, adjusted_probs, adjustment_info)
        """
        adjusted_probs = ml_probabilities.copy()
        adjustments = []
        reasons = []
        
        # Extract key features
        grass_pct = features['grass_coverage']['percentage']
        crack_severity = features['crack_analysis']['severity']
        moisture_level = features['moisture_level']['level']
        color_type = features['color_profile']['color_type']
        
        # Rule 1: High grass → Bowling-friendly
        if grass_pct > 50:
            boost = 0.15
            adjusted_probs[1] += boost  # bowling_friendly
            adjustments.append(f"+{boost:.0%} Bowling-friendly")
            reasons.append(f"High grass coverage ({grass_pct:.1f}%) favors fast bowlers")
        
        # Rule 2: Many cracks → Spin-friendly
        if crack_severity in ['High', 'Severe']:
            boost = 0.20
            adjusted_probs[3] += boost  # spin_friendly
            adjustments.append(f"+{boost:.0%} Spin-friendly")
            reasons.append(f"Severe cracks ({crack_severity}) will assist spinners")
        
        # Rule 3: Dry + Low grass → Seam-friendly
        if moisture_level == 'Dry' and grass_pct < 30:
            boost = 0.15
            adjusted_probs[2] += boost  # seam_friendly
            adjustments.append(f"+{boost:.0%} Seam-friendly")
            reasons.append("Dry pitch with low grass favors seam bowling")
        
        # Rule 4: Wet pitch → Bowling-friendly
        if moisture_level == 'Wet':
            boost = 0.12
            adjusted_probs[1] += boost  # bowling_friendly
            adjustments.append(f"+{boost:.0%} Bowling-friendly")
            reasons.append("Wet pitch assists swing bowling")
        
        # Rule 5: Low cracks + good grass → Batting-friendly
        if crack_severity in ['None', 'Low'] and 20 < grass_pct < 40:
            boost = 0.10
            adjusted_probs[0] += boost  # batting_friendly
            adjustments.append(f"+{boost:.0%} Batting-friendly")
            reasons.append("Minimal cracks and moderate grass = good batting surface")
        
        # Rule 6: Brown/Dark color + low grass → Spin-friendly
        if color_type in ['Brown', 'Dark'] and grass_pct < 25:
            boost = 0.10
            adjusted_probs[3] += boost  # spin_friendly
            adjustments.append(f"+{boost:.0%} Spin-friendly")
            reasons.append("Dark, bare pitch will deteriorate and spin")
        
        # Normalize probabilities
        adjusted_probs = np.maximum(adjusted_probs, 0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        # Get final prediction
        final_idx = adjusted_probs.argmax()
        final_class = self.classes[final_idx]
        final_confidence = adjusted_probs[final_idx] * 100
        
        adjustment_info = {
            "adjustments": adjustments if adjustments else ["No adjustments needed"],
            "reasons": reasons if reasons else ["ML prediction is reliable"]
        }
        
        return final_class, final_confidence, adjusted_probs, adjustment_info
    
    def analyze(self, image_path: str, save_visualization: bool = False) -> Dict:
        """
        Complete pitch analysis pipeline with visualization
        
        Args:
            image_path: Path to input image
            save_visualization: Whether to save visualization
            
        Returns:
            Dictionary with all analysis results including image paths
        """
        print(f"\n{'='*60}")
        print(f"🏏 ANALYZING PITCH IMAGE")
        print(f"{'='*60}\n")
        
        # Load original image for visualization
        original_image = cv2.imread(image_path)
        
        # Step 1: Detect pitch region
        print("📍 Step 1: Detecting pitch region...")
        pitch_region, bbox = self.detect_pitch(image_path)
        
        if pitch_region is None or pitch_region.size == 0:
            raise ValueError("Failed to detect pitch region")
        
        # Verify we have a valid cropped region
        h, w = pitch_region.shape[:2]
        print(f"   ✂️  Using cropped region: {w}x{h} pixels")
        if bbox:
            print(f"   📍 Bounding box: {bbox}")
        
        # Step 2: Extract features FROM CROPPED REGION
        print("\n🔍 Step 2: Extracting pitch features from cropped region...")
        # Save pitch region to temp file for feature analyzer
        import tempfile
        import os
        temp_pitch_path = "temp_pitch_region.jpg"
        cv2.imwrite(temp_pitch_path, pitch_region)
        print(f"   📐 Processing region size: {pitch_region.shape[1]}x{pitch_region.shape[0]} pixels")
        features = self.feature_analyzer.analyze(temp_pitch_path)
        
        # Clean up temp file
        if os.path.exists(temp_pitch_path):
            os.unlink(temp_pitch_path)
        
        # Step 3: ML Classification ON CROPPED REGION
        print("\n🤖 Step 3: Running ML classification on cropped region...")
        print(f"   📐 Classifying region size: {pitch_region.shape[1]}x{pitch_region.shape[0]} pixels")
        ml_class, ml_confidence, ml_probs = self.classify_pitch(pitch_region)
        print(f"   ML Prediction: {ml_class} ({ml_confidence:.1f}%)")
        
        # Step 4: Feature-based adjustment
        print("\n⚖️ Step 4: Adjusting with feature analysis...")
        final_class, final_conf, final_probs, adjustment_info = \
            self.adjust_classification_with_features(ml_probs, features)
        
        print(f"   Final Prediction: {final_class} ({final_conf:.1f}%)")
        
        if adjustment_info['adjustments'][0] != "No adjustments needed":
            print("   Adjustments applied:")
            for adj in adjustment_info['adjustments']:
                print(f"   - {adj}")
        
        # Compile results
        results = {
            'pitch_detection': {
                'detected': bbox is not None,
                'bbox': bbox,
                'confidence': 0.95 if bbox else 0.0
            },
            'features': features,
            'ml_classification': {
                'prediction': ml_class,
                'confidence': ml_confidence,
                'probabilities': ml_probs
            },
            'final_classification': {
                'prediction': final_class,
                'confidence': final_conf,
                'probabilities': final_probs,
                'adjustment_info': adjustment_info
            }
        }
        
        print(f"\n✅ Analysis complete!\n")
        
        return results
    

