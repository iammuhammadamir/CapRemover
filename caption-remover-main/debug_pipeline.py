#!/usr/bin/env python3
"""
Debug script to diagnose why captions aren't being removed.
This will test each stage of the pipeline step by step.
"""

# IMPORTANT: Import torch FIRST to initialize CUDA before PaddlePaddle
import torch
torch.cuda.init()

import cv2
import numpy as np
from decord import VideoReader, cpu
from paddleocr import PaddleOCR
import os

print("=" * 80)
print("CAPTION REMOVER - PIPELINE DEBUGGER")
print("=" * 80)

# Configuration
video_path = "data/examples/long.mp4"
# roi = (145, 700, 600, 180)  # Your current ROI
roi = (230,1300,610,180) # i've set this roi for long.mp4.
frame_idx = 300  # Frame to test

print(f"\nConfiguration:")
print(f"  Video: {video_path}")
print(f"  ROI: {roi}")
print(f"  Test frame: {frame_idx}")
print("")

# Step 1: Check if video exists and can be read
print("[1/5] Checking video file...")
if not os.path.exists(video_path):
    print(f"❌ ERROR: Video file not found: {video_path}")
    exit(1)

vr = VideoReader(video_path, ctx=cpu(0))
print(f"✓ Video loaded: {len(vr)} frames, {vr.get_avg_fps():.2f} fps, {vr[0].shape}")

# Step 2: Extract and save a test frame
print(f"\n[2/5] Extracting test frame {frame_idx}...")
frame_rgb = vr[frame_idx].asnumpy()
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
h, w = frame_bgr.shape[:2]
print(f"✓ Frame extracted: {w}x{h}")

# Save full frame
os.makedirs("data/results/debug", exist_ok=True)
cv2.imwrite("data/results/debug/01_full_frame.jpg", frame_bgr)
print(f"  Saved: data/results/debug/01_full_frame.jpg")

# Step 3: Extract ROI region
print(f"\n[3/5] Extracting ROI region...")
x, y, w_roi, h_roi = roi
roi_frame = frame_bgr[y:y+h_roi, x:x+w_roi].copy()
print(f"✓ ROI extracted: {w_roi}x{h_roi} from position ({x}, {y})")
cv2.imwrite("data/results/debug/02_roi_region.jpg", roi_frame)
print(f"  Saved: data/results/debug/02_roi_region.jpg")

# Draw ROI box on full frame
frame_with_roi = frame_bgr.copy()
cv2.rectangle(frame_with_roi, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 3)
cv2.putText(frame_with_roi, f"ROI: {roi}", (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite("data/results/debug/03_roi_visualization.jpg", frame_with_roi)
print(f"  Saved: data/results/debug/03_roi_visualization.jpg")

# Step 4: Test PaddleOCR text detection on ROI
print(f"\n[4/5] Testing PaddleOCR text detection on ROI region...")
print("  Initializing PaddleOCR (using same method as mask.py)...")

# Use the same method as mask.py
from paddleocr import TextDetection

model = TextDetection(
    model_name="PP-OCRv5_server_det",
    unclip_ratio=2.5,
    limit_side_len=1024
)
print("  Running text detection...")

# Run prediction
result = model.predict([roi_frame], batch_size=1)

# Check results  
print(f"\n  Detection Results:")

# Get detection polygons from result
if result and len(result) > 0:
    pred = result[0]
    dt_polys = pred.json['res']['dt_polys'] if hasattr(pred, 'json') else []
else:
    dt_polys = []

if len(dt_polys) == 0:
    print("  ❌ NO TEXT DETECTED!")
    print("  ")
    print("  Possible reasons:")
    print("    1. ROI is not covering the captions")
    print("    2. Video has no text at this frame")
    print("    3. Text is too small/blurry for detection")
    print("    4. PaddleOCR model issue")
    print("")
    print("  Recommendations:")
    print("    - Check 01_full_frame.jpg and 02_roi_region.jpg to verify captions exist")
    print("    - Adjust ROI to properly cover caption area")
    print("    - Try a different frame_idx where captions are clearer")
else:
    num_boxes = len(dt_polys)
    print(f"  ✓ Detected {num_boxes} text boxes!")
    
    # Draw detected boxes on ROI
    roi_with_boxes = roi_frame.copy()
    for poly in dt_polys:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(roi_with_boxes, [pts], True, (0, 0, 255), 2)
    
    cv2.imwrite("data/results/debug/04_detected_boxes.jpg", roi_with_boxes)
    print(f"  Saved: data/results/debug/04_detected_boxes.jpg (with red boxes)")
    
    # Draw boxes on full frame (offset by ROI position)
    frame_with_detection = frame_bgr.copy()
    for poly in dt_polys:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        pts[:, 0, 0] += x  # Offset x
        pts[:, 0, 1] += y  # Offset y
        cv2.polylines(frame_with_detection, [pts], True, (0, 0, 255), 2)
    
    cv2.imwrite("data/results/debug/05_full_frame_with_detection.jpg", frame_with_detection)
    print(f"  Saved: data/results/debug/05_full_frame_with_detection.jpg")

# Step 5: Test mask creation
print(f"\n[5/5] Creating mask from detected text...")
mask = np.zeros((h, w, 3), np.uint8)

if len(dt_polys) > 0:
    for poly in dt_polys:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        # Offset by ROI position
        pts[:, 0, 0] += x
        pts[:, 0, 1] += y
        cv2.fillPoly(mask, [pts], (255, 255, 255))
    
    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=10)
    
    cv2.imwrite("data/results/debug/06_mask.jpg", mask)
    cv2.imwrite("data/results/debug/07_mask_dilated.jpg", mask_dilated)
    print(f"  ✓ Mask created!")
    print(f"  Saved: data/results/debug/06_mask.jpg")
    print(f"  Saved: data/results/debug/07_mask_dilated.jpg")
    
    # Create overlay
    overlay = frame_bgr.copy()
    overlay_layer = overlay.copy()
    mask_gray = cv2.cvtColor(mask_dilated, cv2.COLOR_BGR2GRAY)
    mask_bool = mask_gray > 0
    overlay_layer[mask_bool] = (0, 255, 0)  # Green
    cv2.addWeighted(overlay_layer, 0.4, overlay, 0.6, 0, overlay)
    cv2.imwrite("data/results/debug/08_overlay.jpg", overlay)
    print(f"  Saved: data/results/debug/08_overlay.jpg (green = mask area)")
else:
    print(f"  ❌ Cannot create mask - no text detected!")

# Summary
print("\n" + "=" * 80)
print("DEBUG SUMMARY")
print("=" * 80)
print(f"\nAll debug images saved to: data/results/debug/")
print(f"\nCheck these files to diagnose the issue:")
print(f"  1. 01_full_frame.jpg - Does this frame have captions?")
print(f"  2. 03_roi_visualization.jpg - Is the green box covering the captions?")
print(f"  3. 02_roi_region.jpg - Is the extracted ROI showing the caption area?")
print(f"  4. 04_detected_boxes.jpg - Are red boxes drawn around detected text?")
print(f"  5. 08_overlay.jpg - Is the green mask covering the caption area?")
print("")

if len(dt_polys) == 0:
    print("⚠️  ACTION REQUIRED:")
    print(f"  1. Check if captions exist in frame {frame_idx}")
    print("  2. Adjust ROI to properly cover caption region")
    print("  3. Try different frame_idx values (e.g., 100, 500, 1000)")
    print("  4. Run this script again after adjustments")
else:
    print("✅ Text detection is working! If captions still aren't removed:")
    print("  1. Make sure you're using the SAME ROI in main.py")
    print("  2. Check that mask_video.mp4 has white regions (not all black)")
    print("  3. Check overlay_video.mp4 has green overlays on captions")
    print("  4. Increase mask_dilation in main.py (try 15-20)")

print("=" * 80)

