#!/usr/bin/env python3
"""
Interactive ROI Finder for Caption Remover
Use this to find the right x, y, width, height for the caption region in your video.

Usage:
    python find_roi.py

Then adjust the values below and run again until the red rectangle perfectly
covers the caption region.
"""

import sys
from src.stages.create_mask import visualize_roi

# ===== CONFIGURATION =====
# Change these values to match your video
video = "data/examples/long.mp4"  # or use the preprocessed version after first run
frame_idx = 300  # Which frame to check (middle of video is usually good)

# ROI format: (x, y, width, height)
# x = horizontal position from left edge
# y = vertical position from top edge  
# width = how wide the caption box is
# height = how tall the caption box is

# ADJUST THESE VALUES:
x = 230      # Start from left edge, increase to move right
y = 1300       # Start from top edge, increase to move down
width = 610   # Make wider to cover more horizontal space
height = 180  # Make taller to cover more vertical space

# for long.mp4 use: (230,1300,610,180)

roi = (x, y, width, height)

# ===== RUN VISUALIZATION =====
print("=" * 60)
print("ROI FINDER")
print("=" * 60)
print(f"Video: {video}")
print(f"Frame: {frame_idx}")
print(f"ROI: x={x}, y={y}, width={width}, height={height}")
print("")
print("The output will be saved to: data/results/roi_preview.png")
print("")
print("Instructions:")
print("1. Check the saved image")
print("2. Adjust x, y, width, height values above if red box doesn't cover captions")
print("3. Run this script again until the box perfectly covers the caption area")
print("4. Copy the final ROI values to main.py line 25")
print("=" * 60)

visualize_roi(video, roi, frame_idx)

print("")
print("Preview saved! Check: data/results/roi_preview.png")
print("")
print("Copy this line to main.py (line 25) when ready:")
print(f"    roi = ({x}, {y}, {width}, {height})")
print("")

