"""
Simple test script for single_frame_pipeline.py

Assumes sample.jpg exists in the current directory.
Runs inference and saves depth.jpg
"""

import logging
from single_frame_pipeline import FlashDepthPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize pipeline
pipeline = FlashDepthPipeline(
    checkpoint_path="configs/flashdepth/iter_43002.pth",  # Update this path
    model_size="vits",
    use_mamba=False,
    device="cuda",
)

# Run inference
depth = pipeline.predict("sample.jpg")

# Save result
pipeline.save_depth_visualization(depth, "depth.jpg")

print(f"Done! Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
