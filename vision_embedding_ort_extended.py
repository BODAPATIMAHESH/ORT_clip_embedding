#!/usr/bin/env python3
"""
Use jina_clip_vision with embedded preprocessing
NOW YOU CAN PASS RAW IMAGE BYTES!
"""

import onnxruntime as ort
from onnxruntime_extensions import get_library_path
import numpy as np
import requests

# ============================================================================
# SETUP
# ============================================================================

print("Setting up ONNX Runtime session...")
so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())

vision_session = ort.InferenceSession(
    "jina_clip_vision_with_preprocessing.onnx",  # NEW MODEL!
    so,
    providers=["CPUExecutionProvider"]
)

print("✓ Model loaded with embedded preprocessing!")

# ============================================================================
# NEW SIMPLIFIED FUNCTIONS
# ============================================================================

def encode_image(image_urls):
    """
    Encode images - NOW MUCH SIMPLER!
    Just download bytes and pass to model!
    """
    # Download all images as bytes
    image_bytes_list = []
    for url in image_urls:
        print(f"Downloading: {url}")
        response = requests.get(url)
        image_bytes_list.append(response.content)
    
    # Convert each to numpy array
    embeddings_list = []
    
    for image_bytes in image_bytes_list:
        # Convert bytes to numpy array
        image_data = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Run inference - preprocessing happens inside the model!
        outputs = vision_session.run(None, {"image_bytes": image_data})
        embeddings_list.append(outputs[0])
    
    # Stack and normalize
    embeddings = np.vstack(embeddings_list)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings

# ============================================================================
# RUN
# ============================================================================

image_urls = [
    "https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg",
    "https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg"
]

print("\nEncoding images...")
image_emb = encode_image(image_urls)

print(f"\n✓ Image embeddings shape: {image_emb.shape}")
print(f"✓ Embeddings:\n{image_emb}")

# Compute similarity
similarity = np.dot(image_emb[0], image_emb[1])
print(f"\n✓ Similarity between images: {similarity:.4f}")

