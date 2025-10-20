"""
Test script for FlashDepthTemporalModel with video input.

Loads sample.pt (a video tensor), runs temporal depth inference, and saves as mp4.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
from flashdepth import FlashDepthTemporalModel

logging.basicConfig(level=logging.INFO)

print("\n" + "="*70)
print("FlashDepthTemporalModel Test with Video")
print("="*70 + "\n")

# Load sample video tensor
print("Loading sample.pt...")
video = torch.load("sample.pt")
print(f"  Video shape: {video.shape}")
print(f"  Video dtype: {video.dtype}")
print(f"  Video range: [{video.min().item():.3f}, {video.max().item():.3f}]")

T, C, H_orig, W_orig = video.shape
assert C == 3, f"Expected 3 channels, got {C}"
print(f"  Video: {T} frames at {W_orig}x{H_orig}")

# Verify input is in [-1, 1] range (or convert if needed)
if video.min() >= 0 and video.max() <= 1:
    print("  Converting from [0, 1] to [-1, 1]...")
    video = video * 2.0 - 1.0
elif video.min() >= 0 and video.max() > 1:
    print("  Converting from [0, 255] to [-1, 1]...")
    video = video / 255.0
    video = video * 2.0 - 1.0

print(f"  Input range after normalization: [{video.min().item():.3f}, {video.max().item():.3f}]")

# Convert to bfloat16 and move to CUDA
video = video.to('cuda').to(torch.bfloat16)

# Initialize temporal model
print("\nInitializing FlashDepthTemporalModel...")
model = FlashDepthTemporalModel(
    model_size='vits',
    checkpoint_path='configs/flashdepth/iter_43002.pth'
)

# Move to CUDA and bfloat16
print("Converting model to CUDA + bfloat16...")
model = model.to('cuda').to(torch.bfloat16)
model.eval()

# Note: Skipping warmup because it interferes with mamba state
# The first forward call will initialize the mamba sequence properly

# Timed inference
print("\nRunning timed inference on full video...")
torch.cuda.synchronize()
import time
start = time.time()

with torch.no_grad():
    depth_output = model(video)

torch.cuda.synchronize()
end = time.time()

print(f"  Inference time: {(end-start):.2f} s")
print(f"  Throughput: {T/(end-start):.2f} FPS")
print(f"  Time per frame: {(end-start)/T*1000:.2f} ms")

# Check output
print("\nOutput statistics:")
print(f"  Shape: {depth_output.shape}")
print(f"  Range: [{depth_output.min().item():.3f}, {depth_output.max().item():.3f}]")
print(f"  Mean: {depth_output.mean().item():.3f}")
print(f"  Dtype: {depth_output.dtype}")

# Verify output
assert depth_output.shape[0] == T, f"Expected {T} frames, got {depth_output.shape[0]}"
assert depth_output.shape[1:] == (H_orig, W_orig), \
    f"Output size {depth_output.shape[1:]} doesn't match input size {(H_orig, W_orig)}"

# Convert depth to video
print("\nConverting depth to video...")
depth_np = depth_output.float().cpu().numpy()  # (T, H, W)

# Convert from [-1, 1] to [0, 1] for visualization
depth_vis = (depth_np + 1.0) / 2.0

# Apply colormap (inferno) to each frame
cmap = cm.get_cmap('inferno')
depth_colored = []
for i in range(T):
    # Apply colormap
    colored = cmap(depth_vis[i])[:, :, :3]  # Drop alpha channel
    # Convert to uint8 BGR for cv2
    colored = (colored * 255).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    depth_colored.append(colored)

# Convert input video for side-by-side visualization
video_np = video.float().cpu().numpy()  # (T, 3, H, W)
video_np = (video_np + 1.0) / 2.0  # Convert to [0, 1]
video_np = (video_np * 255).astype(np.uint8)
video_np = np.transpose(video_np, (0, 2, 3, 1))  # (T, H, W, 3)
video_bgr = []
for i in range(T):
    frame_bgr = cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR)
    video_bgr.append(frame_bgr)

# Create side-by-side video
print("Creating side-by-side video...")
combined_frames = []
for i in range(T):
    combined = np.hstack([video_bgr[i], depth_colored[i]])
    combined_frames.append(combined)

# Save as mp4 at 60fps
output_path = 'depth_causal.mp4'
print(f"\nSaving video to {output_path}...")

fps = 60
height, width = combined_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in combined_frames:
    out.write(frame)

out.release()

# Also save raw depth as numpy array
np.save('depth_causal.npy', depth_np)

print("\n" + "="*70)
print("SUCCESS! Outputs saved:")
print(f"  - {output_path} (side-by-side video at {fps} fps)")
print("  - depth_causal.npy (raw depth values in [-1, 1])")
print("="*70 + "\n")

print("Usage in your pipeline:")
print("-" * 70)
print("""
from flashdepth import FlashDepthTemporalModel

# Initialize and prepare model
model = FlashDepthTemporalModel(
    model_size='vits',
    checkpoint_path='path/to/checkpoint.pth'
)
model = model.to('cuda').to(torch.bfloat16)
model.eval()

# Use in inference
video = ...  # (T, 3, H, W) in [-1, 1] range
with torch.no_grad():
    depth_pred = model(video)  # (T, H, W) in [-1, 1] range
""")
print("="*70 + "\n")
