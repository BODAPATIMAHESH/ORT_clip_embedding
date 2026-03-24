import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import requests
from io import BytesIO

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------
# Load models
# -------------------------
text_session = ort.InferenceSession("jina_clip_text.onnx")
vision_session = ort.InferenceSession("jina_clip_vision.onnx")

# --------------
# inspect what model expects (inputs)
# --------------
print([i.name for i in text_session.get_inputs()])
print([i.name for i in vision_session.get_inputs()])

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-clip-v1",
    trust_remote_code=True
)

# -------------------------
# TEXT ENCODING
# -------------------------
def encode_text(sentences):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="np"
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"],
    }

    outputs = text_session.run(None, ort_inputs)
    embeddings = outputs[0]

    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


# -------------------------
# IMAGE PREPROCESS
# -------------------------
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def preprocess_image(image):
    image = image.resize((224, 224))  # CLIP standard

    img = np.array(image).astype(np.float32) / 255.0

    # CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])

    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    #model expects fp32
    img = img.astype(np.float32)
    return img


def encode_image(image_urls):
    images = [preprocess_image(load_image(url)) for url in image_urls]
    images = np.stack(images)

    ort_inputs = {"pixel_values": images}

    outputs = vision_session.run(None, ort_inputs)
    embeddings = outputs[0]

    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


# -------------------------
# RUN
# -------------------------
sentences = ["A blue cat", "A red cat"]

image_urls = [
    "https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg",
    "https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg"
]

text_emb = encode_text(sentences)
image_emb = encode_image(image_urls)
print(text_emb.shape)
print(image_emb.shape)

# Similarities (dot product since normalized)
print(cosine_similarity(text_emb[0], text_emb[1]))

print(cosine_similarity(text_emb[0], image_emb[0]))
print(cosine_similarity(text_emb[0], image_emb[1]))

print(cosine_similarity(text_emb[1], image_emb[0]))
print(cosine_similarity(text_emb[1], image_emb[1]))

combined_emb = np.concatenate([text_emb, image_emb], axis=1)
combined_emb = combined_emb / np.linalg.norm(combined_emb, axis=1, keepdims=True)
print(combined_emb.shape)
print(cosine_similarity(combined_emb[0], combined_emb[1]))
