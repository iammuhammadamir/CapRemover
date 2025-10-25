import os, cv2, time, queue, threading
import numpy as np
from decord import VideoReader, cpu
from paddleocr import TextDetection

# ============================================================================
# CONFIGURATION - Tune these for optimal performance
# ============================================================================
OUT_MASK = os.path.join("data", "results", "mask_video.mp4")
OUT_OVERLAY = os.path.join("data", "results", "overlay_video.mp4")

DEBUG = True  # Set to True to generate overlay video with visualizations (SLOWER)

# Performance tuning (adjust based on your hardware):
MODEL_NAME = "PP-OCRv5_server_det"  # Use "PP-OCRv5_mobile_det" for 2-3x speed boost
BATCH_SIZE = 32        # Increase for better GPU utilization (try 16, 24, 32)
QUEUE_MAX = 4         # Batches in queue (increase if CPU >> GPU)
LIMIT_SIDE_LEN = 1024   # Lower = faster but less accurate (960, 1280, or None)
                       # For 1080p video: 960 is ~2x faster, 720 is ~3x faster
UNCLIP_RATIO = 2.5     # Expand detected text boxes (higher = larger masks, more stable)
                       # Helps reduce flickering and merges nearby text into one box
                       # Try: 2.0 (moderate), 2.5 (good balance), 3.0+ (very large)

# Visualization settings (only used if DEBUG=True):
OVERLAY_ALPHA = 0.4    # Transparency for overlay fill (0.0 = transparent, 1.0 = opaque)

# Speed optimization tips:
# - Set DEBUG=False for production (skips overlay rendering)
# - Lower LIMIT_SIDE_LEN for 2-3x speedup (960 or 720)
# - Increase BATCH_SIZE until GPU memory is full
# - Use PP-OCRv5_mobile_det instead of server_det for 2-3x faster inference
# - Ensure GPU is being used (check nvidia-smi during run)
# ============================================================================

def draw_outputs(preds, batch_bgr, mask_writer, overlay_writer=None, roi=None, full_h=None, full_w=None, batch_bgr_full=None, mask_dilation=0):
    """Draw detection results on mask and overlay videos"""
    h, w = full_h or batch_bgr.shape[1], full_w or batch_bgr.shape[2]
    for idx, (pred, frame_bgr) in enumerate(zip(preds, batch_bgr)):
        mask = np.zeros((h, w, 3), np.uint8)
        
        # Get detection polygons
        result_data = pred.json['res']
        dt_polys = result_data['dt_polys']
        
        # Draw white filled polygons on mask (offset if ROI)
        for poly in dt_polys:
            pts = np.asarray(poly, np.int32).copy()
            if roi is not None:
                pts[:, 0] += roi[0]  # offset x
                pts[:, 1] += roi[1]  # offset y
            cv2.fillPoly(mask, [pts], (255, 255, 255))
        
        # Apply dilation if specified (expands mask by ~1.5x at 5 iterations)
        if mask_dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=mask_dilation)
        
        mask_writer.write(mask)
        
        # Only create overlay if overlay_writer is provided
        if overlay_writer is not None:
            # Use full frame for overlay if available, otherwise use cropped
            if batch_bgr_full is not None:
                overlay = batch_bgr_full[idx].copy()
            else:
                overlay = frame_bgr.copy()
            
            # Create semi-transparent green overlay using the dilated mask
            overlay_layer = overlay.copy()
            # Convert grayscale mask to boolean for overlay
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bool = mask_gray > 0
            overlay_layer[mask_bool] = (0, 255, 0)
            
            # Also draw original polygon outlines in red for reference
            for poly in dt_polys:
                pts = np.asarray(poly, np.int32).copy()
                if roi is not None:
                    pts[:, 0] += roi[0]
                    pts[:, 1] += roi[1]
                cv2.polylines(overlay_layer, [pts], True, (0, 0, 255), 2)
            
            cv2.addWeighted(overlay_layer, OVERLAY_ALPHA, overlay, 1 - OVERLAY_ALPHA, 0, overlay)
            overlay_writer.write(overlay)

def create_mask_video(video_path, debug=DEBUG, roi=None, mask_dilation=10):
    """
    Create mask video from input video using text detection.
    
    Args:
        video_path: Path to input video (preprocessed or original)
        debug: If True, also creates overlay visualization video
        roi: Optional (x, y, w, h) region to detect text in. If None, detect on full frame.
        mask_dilation: Number of dilation iterations to expand masks (0=no dilation, 5=~1.5x expansion)
    
    Returns:
        Path to generated mask video
    """
    os.makedirs(os.path.dirname(OUT_MASK), exist_ok=True)
    start = time.time()
    
    # Log which video is being used
    print(f"Creating mask from video: {video_path}")
    
    # Initialize video reader
    print("Initializing video reader...")
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)
    h, w = vr[0].shape[:2]
    print(f"Video info: {total} frames, {fps:.2f} fps, {w}x{h}")
    
    # Initialize model and writers
    print("Initializing model and video writers...")
    print(f"Settings: MODEL={MODEL_NAME}, DEBUG={debug}, BATCH_SIZE={BATCH_SIZE}, LIMIT_SIDE_LEN={LIMIT_SIDE_LEN}, UNCLIP_RATIO={UNCLIP_RATIO}, MASK_DILATION={mask_dilation}, ROI={roi}")
    
    # Create model with optimized settings
    model_kwargs = {
        "model_name": MODEL_NAME,
        "unclip_ratio": UNCLIP_RATIO
    }
    if LIMIT_SIDE_LEN is not None:
        model_kwargs["limit_side_len"] = LIMIT_SIDE_LEN
    model = TextDetection(**model_kwargs)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(OUT_MASK, fourcc, fps, (w, h))
    overlay_writer = cv2.VideoWriter(OUT_OVERLAY, fourcc, fps, (w, h)) if debug else None
    
    q = queue.Queue(maxsize=QUEUE_MAX)
    done = object()
    
    # Producer: decode frames -> put RGB batches
    def producer():
        batch_crop = []
        batch_full = [] if debug and roi is not None else None
        for frame in vr:
            rgb_full = frame.asnumpy()  # RGB uint8
            if roi is not None:
                x, y, w_roi, h_roi = roi
                rgb_crop = rgb_full[y:y+h_roi, x:x+w_roi]  # Crop to ROI
                if batch_full is not None:
                    batch_full.append(rgb_full)  # Keep full frame for overlay
            else:
                rgb_crop = rgb_full
            batch_crop.append(rgb_crop)
            if len(batch_crop) == BATCH_SIZE:
                q.put((np.stack(batch_crop, axis=0), 
                       np.stack(batch_full, axis=0) if batch_full else None))
                batch_crop = []
                batch_full = [] if batch_full is not None else None
        if batch_crop:
            q.put((np.stack(batch_crop, axis=0),
                   np.stack(batch_full, axis=0) if batch_full else None))
        q.put(done)
    
    # Consumer: RGB batch -> BGR (vectorized) -> predict -> write
    total_text_detections = 0 # todo: remove this
    def consumer():
        nonlocal total_text_detections
        processed = 0
        while True:
            item = q.get()
            if item is done:
                q.task_done()
                break
            
            batch_crop, batch_full = item
            
            # Vectorized channel swap RGB->BGR; ensure contiguous once
            batch_bgr_crop = np.ascontiguousarray(batch_crop[..., ::-1])
            batch_bgr_full = np.ascontiguousarray(batch_full[..., ::-1]) if batch_full is not None else None
            
            # Predict on batch
            preds = model.predict([batch_bgr_crop[i] for i in range(batch_bgr_crop.shape[0])],
                                  batch_size=batch_bgr_crop.shape[0])
            
            if debug:
                # Count detections for verification
                for pred in preds:
                    result_data = pred.json['res']
                    dt_polys = result_data['dt_polys']
                    total_text_detections += len(dt_polys)
            
            # Draw and write
            draw_outputs(preds, batch_bgr_crop, mask_writer, overlay_writer, roi, h, w, batch_bgr_full, mask_dilation)
            
            processed += batch_bgr_crop.shape[0]
            if processed % 60 == 0 or processed >= total:
                elapsed = time.time() - start
                fps_proc = processed / elapsed
                print(f"Progress: {processed}/{total} frames ({fps_proc:.2f} fps)")
            
            q.task_done()
    
    # Start threads
    print(f"Starting streaming processing (batch size: {BATCH_SIZE})...")
    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)
    t_prod.start()
    t_cons.start()
    t_prod.join()
    q.join()
    
    # Cleanup
    mask_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()
    
    total_time = time.time() - start
    print(f"\n=== Summary ===")
    print(f"Total frames: {total}")
    print(f"Total time: {total_time:.2f}s ({total/total_time:.2f} fps)")
    # todo: remove ths debuggin part
    if debug:
        print(f"Total text detections: {total_text_detections}")
        print(f"Average detections per frame: {total_text_detections/total:.2f}")
        
        # Verification check
        if total_text_detections == 0:
            print(f"\n⚠️  WARNING: NO TEXT DETECTED IN ANY FRAME!")
            print(f"  Possible issues:")
            print(f"    1. ROI is incorrect: {roi}")
            print(f"    2. Video has no text/captions")
            print(f"    3. Text is too small/blurry")
            print(f"  The mask and overlay videos will be blank.")
            print(f"  Check roi_preview.png to verify ROI placement.")
        else:
            print(f"✓ Text detection successful!")
    
    print(f"\nOutput files:")
    print(f"  Mask video: {OUT_MASK}")
    if debug:
        print(f"  Overlay video: {OUT_OVERLAY}")
    
    return OUT_MASK

def visualize_roi(video_path, roi, frame_idx=200, output_path="data/results/roi_preview.png"):
    """Save a preview image showing the ROI box on a specific frame."""
    vr = VideoReader(video_path, ctx=cpu(0))
    frame = vr[frame_idx].asnumpy()  # RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    x, y, w_roi, h_roi = roi
    cv2.rectangle(frame, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 3)
    cv2.putText(frame, f"ROI: {roi}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"ROI preview saved to: {output_path}")

if __name__ == "__main__":
    import sys
    # Add project root to path for imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.stages.postprocessing.compress import compress_videos
    
    # Standalone execution for testing
    # Use the working preprocessed video
    test_video = os.path.join("data", "examples", "34s1080_preprocessed.mp4")
    
    # ROI (x, y, w, h) for the preprocessed video
    test_roi = (150, 880, 600, 200)  # Caption region for 34s1080 preprocessed video
    
    # Visualize ROI first
    visualize_roi(test_video, test_roi, frame_idx=300)
    
    # Create mask with ROI
    mask_video = create_mask_video(test_video, debug=DEBUG, roi=test_roi)
    
    # Compress output videos
    print("\n" + "=" * 60)
    print("COMPRESSING VIDEOS")
    print("=" * 60)
    videos_to_compress = [mask_video]
    if DEBUG:
        videos_to_compress.append(OUT_OVERLAY)
    
    for video_path in videos_to_compress:
        if os.path.exists(video_path):
            temp_path = video_path.replace('.mp4', '_temp.mp4')
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx265',
                '-crf', '23',
                '-preset', 'medium',
                '-y',
                temp_path
            ]
            import subprocess
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(temp_path, video_path)
            print(f"  Compressed: {video_path}")
    
    print("✓ Video compression completed")

