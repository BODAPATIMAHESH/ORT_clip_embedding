import torch
import os
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-large-patch14"
output_path = "clip-vit-large-onnx/model.onnx"
os.makedirs("clip-vit-large-onnx", exist_ok=True)

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
model.eval()

# Wrapper with positional args to avoid dict ordering issues
class CLIPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        return output.logits_per_image, output.logits_per_text, output.text_embeds, output.image_embeds

wrapped_model = CLIPWrapper(model)
wrapped_model.eval()

dummy_inputs = processor(
    text=["a photo of a cat"],
    images=torch.zeros(1, 3, 224, 224),
    return_tensors="pt",
    padding=True,
    do_rescale=False
)

input_ids      = dummy_inputs["input_ids"].long()
attention_mask = dummy_inputs["attention_mask"].long()
pixel_values   = dummy_inputs["pixel_values"].float()

print("input_ids dtype     :", input_ids.dtype)
print("attention_mask dtype:", attention_mask.dtype)
print("pixel_values dtype  :", pixel_values.dtype)

torch.onnx.export(
    wrapped_model,
    (input_ids, attention_mask, pixel_values),
    output_path,
    opset_version=14,
    input_names=["input_ids", "attention_mask", "pixel_values"],
    output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"],
    dynamic_axes={
        "input_ids":      {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "pixel_values":   {0: "batch_size"},
    },
)
print(f"Model exported to {output_path}")
