import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import os

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "google/vit-base-patch16-224"
NUM_CLASSES = 10
ONNX_PATH = "vit_model.onnx"
TEST_IMAGE_URL = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"

# -------------------------------
# Load Pretrained Model + Feature Extractor
# -------------------------------
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True  # 🔧 Fixes classifier size mismatch
)

# -------------------------------
# Download + Preprocess Image
# -------------------------------
image = Image.open(requests.get(TEST_IMAGE_URL, stream=True).raw).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")

# -------------------------------
# Inference
# -------------------------------
with torch.no_grad():
    outputs = model(**inputs).logits
    predicted_class = outputs.argmax(-1).item()
    print(f"Predicted class index: {predicted_class}")

# -------------------------------
# Export to ONNX
# -------------------------------
dummy_input = inputs["pixel_values"]
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=16
)
print(f"✅ ONNX model exported to {ONNX_PATH}")
