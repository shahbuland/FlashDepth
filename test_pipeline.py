"""
Simple test script for single_frame_pipeline.py

Assumes sample.jpg exists in the current directory.
Runs inference with compilation enabled and saves depth.jpg
"""

import logging
from single_frame_pipeline import FlashDepthPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

print("\n" + "="*70)
print("FlashDepth Inference Test (with torch.compile)")
print("="*70 + "\n")

# Initialize pipeline with compilation enabled
print("Initializing pipeline with torch.compile (max-autotune)...")
pipeline = FlashDepthPipeline(
    checkpoint_path="configs/flashdepth/iter_43002.pth",
    model_size="vits",
    use_mamba=False,
    device="cuda",
    compile_model=True,  # Enable compilation
    use_bfloat16=True,   # Use bfloat16 for speed
)

print("\nRunning inference on sample.jpg...")
print("NOTE: First run will be slow due to compilation (this is normal)")
print("-"*70)

# Run inference
depth = pipeline.predict("sample.jpg")

# Save results
print("\nSaving outputs...")
pipeline.save_depth_visualization(depth, "depth.jpg")
pipeline.save_depth_npy(depth, "depth.npy")

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"Depth map shape: {depth.shape}")
print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
print(f"\nOutputs saved:")
print(f"  - depth.jpg (visualization)")
print(f"  - depth.npy (raw depth values)")
print("="*70 + "\n")
