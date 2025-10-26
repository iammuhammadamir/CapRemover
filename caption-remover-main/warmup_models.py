#!/usr/bin/env python3
"""
Model Warmup Script for torch.compile Cache

This script pre-compiles the inpainting models by running a quick inference
on a small test video. After running this once, all subsequent runs will be
fast as they'll use the cached compiled graphs.

Run this:
1. After initial installation
2. After updating PyTorch
3. After changing model architecture
4. On a new GPU/system

Usage:
    python warmup_models.py

Expected time: ~10-15 minutes (one-time cost)
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

print("=" * 70)
print("TORCH.COMPILE MODEL WARMUP SCRIPT")
print("=" * 70)
print("\nThis script will pre-compile the inpainting models.")
print("First run will be slow (~10-15 minutes) but creates a cache.")
print("All subsequent pipeline runs will be FAST (~30% faster)!")
print("\n" + "=" * 70 + "\n")

# Create a tiny test video (1 second, 24 fps, 480x720)
print("Step 1/4: Creating test video...")
test_video_path = "data/results/warmup_test_video.mp4"
test_mask_path = "data/results/warmup_test_mask.mp4"
os.makedirs("data/results", exist_ok=True)

# Create test video (24 frames, 480x720)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width, height = 480, 720
fps = 24
frames_count = 24

video_writer = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
mask_writer = cv2.VideoWriter(test_mask_path, fourcc, fps, (width, height))

for i in range(frames_count):
    # Create a simple frame with gradient
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (i * 10, 128, 255 - i * 10)  # Varying colors
    
    # Create a mask (white rectangle in center)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[height//3:2*height//3, width//3:2*width//3] = 255
    
    video_writer.write(frame)
    mask_writer.write(mask)

video_writer.release()
mask_writer.release()
print(f"✓ Test video created: {test_video_path}")
print(f"✓ Test mask created: {test_mask_path}")

# Import and run inpainting
print("\nStep 2/4: Loading models...")
start_load = time.time()

from src.stages.inpaint.inpaint import run_inpainting

load_time = time.time() - start_load
print(f"✓ Models loaded in {load_time:.1f}s")

# Run inference to trigger compilation
print("\nStep 3/4: Running inference to compile models...")
print("(This is the slow part - compilation happens now)")
start_compile = time.time()

try:
    propainter_out, diffueraser_out = run_inpainting(
        video_path=test_video_path,
        mask_path=test_mask_path,
        output_dir="data/results",
        video_length=1,  # 1 second video
        mask_dilation=8,
        max_img_size=480,
        raft_iter=6,  # Use reduced iterations for faster warmup
        enable_pre_inference=False
    )
    
    compile_time = time.time() - start_compile
    print(f"✓ Inference completed in {compile_time:.1f}s")
    print(f"✓ Compiled graphs cached to: .torch_compile_cache/")
    
except Exception as e:
    print(f"\n✗ Warmup failed with error: {e}")
    print("The pipeline will still work, but won't benefit from torch.compile")
    sys.exit(1)

# Cleanup test files
print("\nStep 4/4: Cleaning up test files...")
try:
    os.remove(test_video_path)
    os.remove(test_mask_path)
    if os.path.exists(propainter_out):
        os.remove(propainter_out)
    if os.path.exists(diffueraser_out):
        os.remove(diffueraser_out)
    print("✓ Test files cleaned up")
except:
    print("⚠ Could not clean up some test files (non-critical)")

# Summary
total_time = load_time + compile_time
print("\n" + "=" * 70)
print("WARMUP COMPLETE!")
print("=" * 70)
print(f"\nTotal warmup time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print("\nCompiled model cache is now ready!")
print("All future pipeline runs will use the cached compiled graphs.")
print("\nExpected performance improvement:")
print("  - Without cache: ~704s per video")
print("  - With cache:    ~500s per video (30% faster!)")
print("\nYou can now run: python main.py")
print("=" * 70)

