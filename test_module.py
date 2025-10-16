"""
Test script for FlashDepthModel nn.Module wrapper.

Loads sample.jpg, converts to [-1, 1] bfloat16, runs inference, and saves depth map.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from flashdepth import FlashDepthModel

logging.basicConfig(level=logging.INFO)

print("\n" + "="*70)
print("FlashDepthModel Test with Real Image")
print("="*70 + "\n")

# Load sample image
print("Loading sample.jpg...")
image = cv2.imread("sample.jpg")
if image is None:
    raise FileNotFoundError("sample.jpg not found!")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H_orig, W_orig = image.shape[:2]
print(f"  Original size: {W_orig}x{H_orig}")

# Convert to tensor and normalize to [-1, 1]
print("\nConverting to tensor and normalizing to [-1, 1]...")
image_tensor = torch.from_numpy(image).float() / 255.0  # [0, 1]
image_tensor = image_tensor * 2.0 - 1.0  # [-1, 1]
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

# Convert to bfloat16 and move to CUDA
image_tensor = image_tensor.to('cuda').to(torch.bfloat16)

print(f"  Input shape: {image_tensor.shape}")
print(f"  Input range: [{image_tensor.min().item():.3f}, {image_tensor.max().item():.3f}]")
print(f"  Input dtype: {image_tensor.dtype}")

# Initialize model
print("\nInitializing FlashDepthModel...")
model = FlashDepthModel(
    model_size='vits',
    use_mamba=False,
    checkpoint_path='configs/flashdepth/iter_43002.pth'
)

# Move to CUDA and bfloat16
print("Converting model to CUDA + bfloat16...")
model = model.to('cuda').to(torch.bfloat16)
model.eval()

# Compile the inference function
print("Compiling inference function with torch.compile (mode=max-autotune)...")
print("NOTE: First run will be slow due to compilation")
model.compile(mode='max-autotune', dynamic=False)

# Warmup run
print("\n" + "-"*70)
print("Running warmup inference (compilation happens here)...")
with torch.no_grad():
    _ = model(image_tensor)
print("Warmup complete!")

# Timed inference
print("\nRunning timed inference...")
torch.cuda.synchronize()
import time
start = time.time()

with torch.no_grad():
    depth_output = model(image_tensor)

torch.cuda.synchronize()
end = time.time()

print(f"  Inference time: {(end-start)*1000:.2f} ms")
print(f"  Throughput: {1.0/(end-start):.2f} FPS")

# Check output
print("\nOutput statistics:")
print(f"  Shape: {depth_output.shape}")
print(f"  Range: [{depth_output.min().item():.3f}, {depth_output.max().item():.3f}]")
print(f"  Mean: {depth_output.mean().item():.3f}")
print(f"  Dtype: {depth_output.dtype}")

# Verify output is in [-1, 1] range
assert depth_output.min() >= -1.0, "Output minimum is below -1"
assert depth_output.max() <= 1.0, "Output maximum is above 1"

# Verify output has same spatial size as input
assert depth_output.shape[-2:] == (H_orig, W_orig), \
    f"Output size {depth_output.shape[-2:]} doesn't match input size {(H_orig, W_orig)}"

# Convert depth to visualization
print("\nConverting depth to visualization...")
depth_np = depth_output.squeeze(0).float().cpu().numpy()  # (H, W)

# Convert from [-1, 1] to [0, 1] for visualization
depth_vis = (depth_np + 1.0) / 2.0

# Apply colormap
plt.figure(figsize=(depth_vis.shape[1] / 100, depth_vis.shape[0] / 100), dpi=100)
plt.imshow(depth_vis, cmap='inferno')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('depth_module.jpg', bbox_inches='tight', pad_inches=0, dpi=100)
plt.close()

# Also save raw depth
np.save('depth_module.npy', depth_np)

print("\n" + "="*70)
print("SUCCESS! Outputs saved:")
print("  - depth_module.jpg (visualization)")
print("  - depth_module.npy (raw depth values in [-1, 1])")
print("="*70 + "\n")

print("Usage in your training pipeline:")
print("-" * 70)
print("""
from flashdepth import FlashDepthModel

# Initialize and prepare model
model = FlashDepthModel(
    model_size='vits',
    checkpoint_path='path/to/checkpoint.pth'
)
model = model.to('cuda').to(torch.bfloat16)
model = torch.compile(model, mode='max-autotune')

# Use in training loop
images = ...  # (B, 3, H, W) in [-1, 1] range
with torch.no_grad():
    depth_pred = model(images)  # (B, H, W) in [-1, 1] range

# For training with gradients, just remove torch.no_grad()
loss = criterion(depth_pred, depth_gt)
loss.backward()
""")
print("="*70 + "\n")
