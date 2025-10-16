"""
Speed test script for FlashDepth pipeline.

Tests inference speed on 360p frames (640x360) with batch size of 4.
Runs warmup iterations followed by timed evaluation.
"""

import logging
import time
import torch
import numpy as np
from single_frame_pipeline import FlashDepthPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_dummy_frames(batch_size=4, height=360, width=640):
    """
    Create dummy RGB frames for testing.

    Args:
        batch_size: Number of frames
        height: Frame height
        width: Frame width

    Returns:
        List of numpy arrays (H, W, 3) in float32 [0, 1]
    """
    frames = []
    for i in range(batch_size):
        # Create random RGB image
        frame = np.random.rand(height, width, 3).astype(np.float32)
        frames.append(frame)
    return frames

def run_speed_test(use_compile=False, use_bfloat16=True):
    """
    Run speed test with specified configuration.

    Args:
        use_compile: Whether to use torch.compile
        use_bfloat16: Whether to use bfloat16 precision
    """
    print("\n" + "="*70)
    print(f"SPEED TEST: compile={use_compile}, bfloat16={use_bfloat16}")
    print("="*70)

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = FlashDepthPipeline(
        checkpoint_path="configs/flashdepth/iter_43002.pth",
        model_size="vits",
        use_mamba=False,
        device="cuda",
        compile_model=use_compile,
        use_bfloat16=use_bfloat16,
    )

    # Create dummy frames (360p resolution, batch of 4)
    batch_size = 4
    height = 360
    width = 640
    print(f"\nGenerating {batch_size} dummy frames at {width}x{height}...")
    frames = create_dummy_frames(batch_size, height, width)

    # Warmup runs
    num_warmup = 10
    print(f"\nRunning {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        for frame in frames:
            _ = pipeline.predict(frame, return_numpy=True)
        if (i + 1) % 5 == 0:
            print(f"  Warmup {i+1}/{num_warmup} complete")

    # Sync GPU before timing
    torch.cuda.synchronize()

    # Evaluation runs
    num_eval = 10
    print(f"\nRunning {num_eval} timed evaluation iterations...")
    latencies = []

    for i in range(num_eval):
        # Time a single batch
        torch.cuda.synchronize()
        start_time = time.time()

        for frame in frames:
            _ = pipeline.predict(frame, return_numpy=True)

        torch.cuda.synchronize()
        end_time = time.time()

        batch_time = end_time - start_time
        latencies.append(batch_time)

        if (i + 1) % 5 == 0:
            print(f"  Eval {i+1}/{num_eval} complete")

    # Calculate statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    # Per-frame statistics
    mean_per_frame = mean_latency / batch_size
    fps_per_frame = 1.0 / mean_per_frame

    # Print results
    print("\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"Configuration:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Compile: {use_compile}")
    print(f"  - BFloat16: {use_bfloat16}")
    print(f"\nBatch Statistics (processing {batch_size} frames):")
    print(f"  - Mean time: {mean_latency*1000:.2f} ms")
    print(f"  - Std dev: {std_latency*1000:.2f} ms")
    print(f"  - Min time: {min_latency*1000:.2f} ms")
    print(f"  - Max time: {max_latency*1000:.2f} ms")
    print(f"\nPer-Frame Statistics:")
    print(f"  - Mean latency: {mean_per_frame*1000:.2f} ms/frame")
    print(f"  - Throughput: {fps_per_frame:.2f} FPS")
    print(f"  - Total throughput: {fps_per_frame * batch_size:.2f} frames/sec (batch)")
    print("="*70 + "\n")

    return {
        'mean_latency': mean_latency,
        'mean_per_frame': mean_per_frame,
        'fps': fps_per_frame,
        'batch_size': batch_size,
    }

if __name__ == "__main__":
    print("\nFlashDepth Pipeline Speed Test")
    print("Testing on 360p frames (640x360) with batch size 4")
    print("="*70)

    results = {}

    # Test 1: No compile, with bfloat16
    results['no_compile'] = run_speed_test(use_compile=False, use_bfloat16=True)

    # Test 2: With compile, with bfloat16
    results['with_compile'] = run_speed_test(use_compile=True, use_bfloat16=True)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)

    speedup = results['no_compile']['mean_per_frame'] / results['with_compile']['mean_per_frame']

    print(f"Without compile: {results['no_compile']['mean_per_frame']*1000:.2f} ms/frame ({results['no_compile']['fps']:.2f} FPS)")
    print(f"With compile:    {results['with_compile']['mean_per_frame']*1000:.2f} ms/frame ({results['with_compile']['fps']:.2f} FPS)")
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Improvement: {(speedup-1)*100:.1f}%")
    print("="*70 + "\n")
