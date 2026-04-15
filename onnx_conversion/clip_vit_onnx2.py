import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor

model_name = "openai/clip-vit-large-patch14"
onnx_path  = "clip-vit-large-onnx/model.onnx"

# ---- Load processor & session ----
processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
session   = ort.InferenceSession(onnx_path)

# ---- Verify ONNX input types ----
print("Expected ONNX inputs:")
for inp in session.get_inputs():
    print(f"  name: {inp.name}, dtype: {inp.type}, shape: {inp.shape}")

# ---- Load image ----
url   = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# ---- Prepare inputs ----
texts  = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# ✅ Cast to correct dtypes
onnx_inputs = {
    "input_ids":      inputs["input_ids"].cpu().numpy().astype(np.int64),
    "attention_mask": inputs["attention_mask"].cpu().numpy().astype(np.int64),
    "pixel_values":   inputs["pixel_values"].cpu().numpy().astype(np.float32),
}

# ---- Run inference ----
outputs      = session.run(None, onnx_inputs)
output_names = [o.name for o in session.get_outputs()]
output_dict  = dict(zip(output_names, outputs))

# ---- Print embeddings ----
if "image_embeds" in output_dict:
    print("\nImage Embeddings Shape:", output_dict["image_embeds"].shape)

if "text_embeds" in output_dict:
    print("Text Embeddings Shape :", output_dict["text_embeds"].shape)

# ---- Softmax (same as logits_per_image.softmax(dim=1)) ----
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

logits_per_image = output_dict.get("logits_per_image", outputs[0])
probs = softmax(logits_per_image)

print("\nLabel Probabilities:")
for text, prob in zip(texts, probs[0]):
    print(f"  {text}: {prob:.4f}")
