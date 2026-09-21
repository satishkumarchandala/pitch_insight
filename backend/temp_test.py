import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import onnxruntime as ort

# 1. Setup PT
try:
    classifier_pt = models.mobilenet_v2(weights=None)
    classifier_pt.classifier[1] = torch.nn.Linear(classifier_pt.classifier[1].in_features, 4)
    state_dict = torch.load("best_pitch_classifier.pth", map_location='cpu')
    try:
        classifier_pt.load_state_dict(state_dict)
    except:
        classifier_pt = models.resnet18()
        classifier_pt.fc = torch.nn.Linear(classifier_pt.fc.in_features, 4)
        classifier_pt.load_state_dict(state_dict)
    classifier_pt.eval()
    pt_loaded = True
except Exception as e:
    print("PT load error:", e)
    pt_loaded = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Setup ONNX
try:
    classifier_onnx = ort.InferenceSession("pitch_classifier.onnx", providers=['CPUExecutionProvider'])
    onnx_input_name = classifier_onnx.get_inputs()[0].name
    onnx_loaded = True
except Exception as e:
    print("ONNX load error:", e)
    onnx_loaded = False

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def preprocess_pt(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return transform(pil_image).unsqueeze(0)

def preprocess_onnx(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_float = image_resized.astype(np.float32) / 255.0
    image_chw = np.transpose(image_float, (2, 0, 1))
    image_batch = np.expand_dims(image_chw, axis=0)
    image_normalized = (image_batch - mean) / std
    return image_normalized.astype(np.float32)


# Create dummy image
img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)

print("----- Raw Output Probs -----")
# run PT
if pt_loaded:
    with torch.no_grad():
        out_pt = classifier_pt(preprocess_pt(img))
        probs_pt = torch.nn.functional.softmax(out_pt[0], dim=0).numpy()
    print("PT:", probs_pt)

# run ONNX
if onnx_loaded:
    out_onnx = classifier_onnx.run(None, {onnx_input_name: preprocess_onnx(img)})
    logits = out_onnx[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs_onnx = exp_logits / exp_logits.sum()
    print("ONNX:", probs_onnx)
