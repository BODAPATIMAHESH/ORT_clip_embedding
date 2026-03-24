import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor
import torch
import time

# ---- Load ONNX model ----
session = ort.InferenceSession(
        "/home/maheshbodapati/openai-clip-vit-onnx/model.onnx",
        providers=["CPUExecutionProvider"]
        )

# ---- Load processor (still needed for tokenization + image preprocessing) ----
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load image ----
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# ---- Prepare inputs ----
inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True
        )

# Convert torch tensors → numpy
onnx_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
        "pixel_values": inputs["pixel_values"].cpu().numpy(),
        }
start = time.time()
# ---- Run inference ----
outputs = session.run(None, onnx_inputs)
end = time.time()
print("Inference time:", (end - start) * 1000, "ms")
# Map outputs by name (safer)
output_names = [o.name for o in session.get_outputs()]
output_dict = dict(zip(output_names, outputs))

# ---- Print embeddings ----
if "image_embeds" in output_dict:
    print("\nImage Embeddings Shape:", output_dict["image_embeds"].shape)
    print(output_dict["image_embeds"])

if "text_embeds" in output_dict:
    print("\nText Embeddings Shape:", output_dict["text_embeds"].shape)
    print(output_dict["text_embeds"])

if "logits_per_image" in output_dict:
    print("\nLogits Per Image Shape:", output_dict["logits_per_image"].shape)
    print(output_dict["logits_per_image"])

if "logits_per_text" in output_dict:
    print("\nLogits Per Text Shape:", output_dict["logits_per_text"].shape)
    print(output_dict["logits_per_text"])

# Depending on export, outputs order may differ.
# Usually CLIP returns:
# 0 = logits_per_image
# 1 = logits_per_text

logits_per_image = outputs[0]

# ---- Softmax (NumPy version) ----
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

probs = softmax(logits_per_image)

print("Probabilities:", probs)
