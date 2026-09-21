import numpy as np
import torch
from torchvision import models
import onnxruntime as ort

print("1. Loading Models...")
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

classifier_onnx = ort.InferenceSession("pitch_classifier.onnx", providers=['CPUExecutionProvider'])
onnx_input_name = classifier_onnx.get_inputs()[0].name

print("2. Generating test tensor...")
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

print("3. Running PT...")
with torch.no_grad():
    pt_output = classifier_pt(torch.from_numpy(dummy_input)).numpy()

print("4. Running ONNX...")
onnx_output = classifier_onnx.run(None, {onnx_input_name: dummy_input})[0]

print("PT Logits:  ", pt_output)
print("ONNX Logits:", onnx_output)

difference = np.abs(pt_output - onnx_output).max()
print("Max Diff on Same Input Tensor:", difference)
