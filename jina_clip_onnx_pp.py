#!/usr/bin/env python3
"""
Add preprocessing to jina_clip_vision.onnx
This will allow you to pass raw image bytes directly!
"""

import onnx
from onnxruntime_extensions.tools.pre_post_processing import (
    PrePostProcessor,
    ConvertImageToBGR,
    ReverseAxis,
    Resize,
    CenterCrop,
    ImageBytesToFloat,
    ChannelsLastToChannelsFirst,
    Normalize,
    Unsqueeze,
    create_named_value
)


def add_preprocessing_to_vision_model(input_model, output_model):
    """
    Add CLIP vision preprocessing to jina_clip_vision.onnx
    """
    
    print(f"Loading model: {input_model}")
    model = onnx.load(input_model)
    
    # Create new input for raw image bytes
    new_input = create_named_value('image_bytes', onnx.TensorProto.UINT8, ['num_bytes'])
    
    # Create preprocessing pipeline
    pipeline = PrePostProcessor([new_input], onnx_opset=18)
    
    # Add CLIP vision preprocessing steps
    pipeline.add_pre_processing([
        ConvertImageToBGR(),              # Decode JPG/PNG to BGR
        ReverseAxis(axis=2, dim_value=3, name="BGR_to_RGB"),  # BGR to RGB
        Resize(224),                       # Resize to 224
        CenterCrop(224, 224),             # Center crop 224x224
        ImageBytesToFloat(1.0/255.0),     # Scale to [0, 1]
        ChannelsLastToChannelsFirst(),    # HWC to CHW
        Normalize(                        # CLIP normalization
            [(0.48145466, 0.26862954),    # R: mean, std
             (0.4578275, 0.26130258),     # G: mean, std
             (0.40821073, 0.27577711)],   # B: mean, std
            layout="CHW"
        ),
        Unsqueeze([0])                    # Add batch dimension
    ])
    
    # Create updated model
    print("Creating model with preprocessing...")
    new_model = pipeline.run(model)
    
    # Save
    print(f"Saving to: {output_model}")
    onnx.save_model(new_model, output_model)
    print("✓ Model with preprocessing created successfully!")
    
    # Verify
    print("\nVerifying model...")
    onnx.checker.check_model(new_model)
    print("✓ Model is valid!")


if __name__ == "__main__":
    # Update these paths
    input_model = "jina_clip_vision.onnx"
    output_model = "jina_clip_vision_with_preprocessing.onnx"
    
    add_preprocessing_to_vision_model(input_model, output_model)
    print("✓ Done! Now We can use the new model with raw image bytes!")

