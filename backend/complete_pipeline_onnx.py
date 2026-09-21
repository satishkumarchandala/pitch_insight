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
from torchvision import transforms


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
        
        # Image transform for classifier - IDENTICAL to complete_pipeline.py
        # Uses PIL-based bilinear resize with anti-aliasing + ImageNet normalisation
        self._classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
                                 scale: float, padding: Tuple[int, int], conf_threshold: float = 0.5) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Postprocess YOLO ONNX output to get bounding boxes using NMS.

        Args:
            output: Raw YOLO output
            original_shape: Original image shape (h, w)
            scale: Scale factor used in preprocessing
            padding: Padding offsets (left, top)
            conf_threshold: Confidence threshold

        Returns:
            None or (bbox, confidence) where bbox is (x1, y1, x2, y2)
        """
        # YOLOv8 output format: [1, 84, 8400] or [1, 8400, 84]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension

        # Ensure shape is [num_detections, features]
        if output.shape[0] < output.shape[1]:
            output = output.T

        # Extract boxes [cx, cy, w, h] and confidences
        if output.shape[1] >= 5:
            boxes_cxcywh = output[:, :4]  # in 640x640 padded space
            confidences = output[:, 4]
        else:
            boxes_cxcywh = output[:, :4]
            confidences = output[:, 4:].max(axis=1) if output.shape[1] > 4 else np.ones(len(output))

        # Pre-filter by confidence
        mask = confidences >= conf_threshold
        if not mask.any():
            print(f"   No detections above threshold {conf_threshold}")
            return None

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]

        # Convert cx,cy,w,h → x1,y1,w,h for NMS (still in padded 640x640 space)
        nms_boxes = []
        for cx, cy, bw, bh in boxes_cxcywh:
            x1 = float(cx - bw / 2)
            y1 = float(cy - bh / 2)
            nms_boxes.append([x1, y1, float(bw), float(bh)])

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confidences.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=0.45
        )

        if len(indices) == 0:
            print("   No detections survived NMS")
            return None

        # Flatten indices (OpenCV returns nested list in some versions)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

        # Pick highest-confidence box after NMS
        best_idx = indices[np.argmax(confidences[indices])]
        best_box_cxcywh = boxes_cxcywh[best_idx]
        confidence = float(confidences[best_idx])

        cx, cy, bw, bh = best_box_cxcywh

        # Remove letterbox padding offsets BEFORE scaling back to original image
        left_pad, top_pad = padding
        cx_no_pad = cx - left_pad
        cy_no_pad = cy - top_pad

        # Scale back to original image coordinates
        cx_orig = cx_no_pad / scale
        cy_orig = cy_no_pad / scale
        bw_orig = bw / scale
        bh_orig = bh / scale

        # Convert to corner format
        x1 = int(cx_orig - bw_orig / 2)
        y1 = int(cy_orig - bh_orig / 2)
        x2 = int(cx_orig + bw_orig / 2)
        y2 = int(cy_orig + bh_orig / 2)

        # Clip to image bounds
        h, w = original_shape
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        print(f"   Debug: Best box after NMS (padded space): cx={cx:.1f}, cy={cy:.1f}, w={bw:.1f}, h={bh:.1f}, conf={confidence:.4f}")
        print(f"   Debug: Final bbox (original image): ({x1}, {y1}, {x2}, {y2})")

        return (x1, y1, x2, y2), confidence
    
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
        Preprocess image for classifier ONNX model.

        Uses torchvision.transforms pipeline identical to the .pt pipeline so
        that the input tensor fed to the ONNX model is bit-for-bit the same
        as the one fed to the PyTorch model (PIL-based bilinear with anti-aliasing,
        then ImageNet normalisation).

        Args:
            image: BGR image from OpenCV

        Returns:
            Preprocessed float32 NCHW tensor ready for ONNX Runtime
        """
        # Convert BGR → RGB and wrap in PIL (matches torchvision pipeline exactly)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Apply the SAME transform chain used by complete_pipeline.py
        img_tensor = self._classifier_transform(pil_image)  # shape: [3, 224, 224]

        # Add batch dimension and convert to numpy float32 for ONNX Runtime
        return img_tensor.unsqueeze(0).numpy().astype(np.float32)
    
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
        Adjust ML classification based on extracted features using cricket logic.

        Synchronized with complete_pipeline.py — uses an accumulator dict so all
        rules fire independently before a single normalisation pass, preventing
        sequential mutations from skewing earlier rule effects.

        Cricket Pitch Rules:
        - High grass coverage → More swing → Bowling-friendly (helps fast bowlers)
        - Many cracks → More spin → Spin-friendly (helps spinners)
        - Dry pitch + low grass → Seam movement → Seam-friendly (helps seamers)
        - Low cracks + moderate grass → Easier batting → Batting-friendly
        - Very dry + many cracks → Strong spin-friendly
        - Wet/Damp + High grass → Strong bowling-friendly

        Args:
            ml_probabilities: Original ML probabilities
            features: Extracted pitch features

        Returns:
            (final_class, confidence, adjusted_probs, adjustment_info)
        """
        # Start with ML probabilities; collect per-class adjustments separately
        adjusted_probs = ml_probabilities.copy()
        adjustments = {
            'batting_friendly': 0.0,
            'bowling_friendly': 0.0,
            'seam_friendly': 0.0,
            'spin_friendly': 0.0
        }
        reasons = []

        # Extract feature values
        grass_pct       = features['grass_coverage']['percentage']
        crack_severity  = features['crack_analysis']['severity']
        crack_density   = features['crack_analysis']['density']
        moisture_level  = features['moisture_level']['level']
        moisture_score  = features['moisture_level']['score']

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
                reasons.append("Moderate cracking helps spin bowlers")

        # RULE 3: Dry Pitch + Low Grass → Seam-friendly
        if moisture_level in ['Dry', 'Slightly Damp'] and grass_pct < 30:
            adjustment = 0.15
            adjustments['seam_friendly'] += adjustment
            reasons.append(f"Dry pitch ({moisture_level}) with minimal grass favors seamers")

        # RULE 4: Low Cracks + Moderate Grass → Batting-friendly
        if crack_severity in ['None', 'Low'] and 20 < grass_pct < 50:
            adjustment = 0.10
            adjustments['batting_friendly'] += adjustment
            reasons.append("Minimal cracks with moderate grass favors batsmen")

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

        # Apply all accumulated adjustments in one pass (prevents sequential skew)
        for i, cls in enumerate(self.classes):
            adjusted_probs[i] += adjustments[cls]

        # Normalize probabilities
        adjusted_probs = np.maximum(adjusted_probs, 0)
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

        # Print active adjustments (adjustments is now a dict {class: delta})
        active_adjustments = {cls: val for cls, val in adjustment_info['adjustments'].items() if val > 0}
        if active_adjustments:
            print("   Adjustments applied:")
            for cls, val in active_adjustments.items():
                print(f"   - +{val:.0%} {cls.replace('_', ' ').title()}")
        else:
            print("   No feature-based adjustments applied")

        if adjustment_info['reasons']:
            print("   Reasons:")
            for reason in adjustment_info['reasons']:
                print(f"   • {reason}")

        # Convert adjustments dict → list of human-readable strings for API response
        adjustments_list = (
            [f"+{val:.0%} {cls.replace('_', ' ').title()}" for cls, val in adjustment_info['adjustments'].items() if val > 0]
            or ["No adjustments needed"]
        )

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
                'confidence': float(ml_confidence),
                'probabilities': {
                    self.classes[i]: float(ml_probs[i] * 100)
                    for i in range(len(self.classes))
                }
            },
            'final_classification': {
                'prediction': final_class,
                'confidence': float(final_conf),
                'probabilities': {
                    self.classes[i]: float(final_probs[i] * 100)
                    for i in range(len(self.classes))
                },
                'adjustments': adjustments_list,
                'reasons': adjustment_info['reasons'] if adjustment_info['reasons'] else ["ML prediction is reliable"]
            }
        }

        print(f"\n✅ Analysis complete!\n")

        return results
    

