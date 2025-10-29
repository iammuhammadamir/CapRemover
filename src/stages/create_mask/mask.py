import os, cv2, time, queue, threading
import numpy as np
from decord import VideoReader, cpu
from paddleocr import TextDetection

# ============================================================================
# CONFIGURATION - Tune these for optimal performance
# ============================================================================
OUT_MASK = os.path.join("data", "results", "mask_video.mp4")
OUT_OVERLAY = os.path.join("data", "results/overlay_video.mp4")

DEBUG = True  # Set to True to generate overlay video with visualizations (SLOWER)

# Performance tuning (adjust based on your hardware):
MODEL_NAME = "PP-OCRv5_server_det"
BATCH_SIZE = 32
QUEUE_MAX = 4
LIMIT_SIDE_LEN = 1024
UNCLIP_RATIO = 2.5

# Visualization settings (only used if DEBUG=True):
OVERLAY_ALPHA = 0.4

def create_single_mask(pred, h, w, roi, mask_dilation):
    mask = np.zeros((h, w, 3), np.uint8)
    result_data = pred.json['res']
    dt_polys = result_data['dt_polys']
    for poly in dt_polys:
        pts = np.asarray(poly, np.int32).copy()
        if roi is not None:
            pts[:, 0] += roi[0]
            pts[:, 1] += roi[1]
        cv2.fillPoly(mask, [pts], (255, 255, 255))
    if mask_dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=mask_dilation)
    return mask, dt_polys

def create_single_overlay(frame_bgr, mask, dt_polys, roi):
    overlay = frame_bgr.copy()
    overlay_layer = overlay.copy()
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bool = mask_gray > 0
    overlay_layer[mask_bool] = (0, 255, 0)
    for poly in dt_polys:
        pts = np.asarray(poly, np.int32).copy()
        if roi is not None:
            pts[:, 0] += roi[0]
            pts[:, 1] += roi[1]
        cv2.polylines(overlay_layer, [pts], True, (0, 0, 255), 2)
    cv2.addWeighted(overlay_layer, OVERLAY_ALPHA, overlay, 1 - OVERLAY_ALPHA, 0, overlay)
    return overlay

def draw_outputs(preds, batch_bgr, mask_writer, overlay_writer=None, roi=None, full_h=None, full_w=None, batch_bgr_full=None, mask_dilation=0):
    h, w = full_h or batch_bgr.shape[1], full_w or batch_bgr.shape[2]
    for idx, (pred, frame_bgr) in enumerate(zip(preds, batch_bgr)):
        mask, dt_polys = create_single_mask(pred, h, w, roi, mask_dilation)
        mask_writer.write(mask)
        if overlay_writer is not None:
            if batch_bgr_full is not None:
                frame_for_overlay = batch_bgr_full[idx]
            else:
                frame_for_overlay = frame_bgr
            overlay = create_single_overlay(frame_for_overlay, mask, dt_polys, roi)
            overlay_writer.write(overlay)

def create_mask_video(video_path, debug=DEBUG, roi=None, mask_dilation=10):
    os.makedirs(os.path.dirname(OUT_MASK), exist_ok=True)
    start = time.time()
    print(f"mask.create_mask_video: Creating mask from video: {video_path}")
    print(f"mask.create_mask_video: OPTIMIZATION ENABLED - Processing only even frames (50% OCR reduction)")
    print(f"mask.create_mask_video: Initializing video reader...")
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)
    h, w = vr[0].shape[:2]
    print(f"mask.create_mask_video: Video info: {total} frames, {fps:.2f} fps, {w}x{h}")
    print(f"mask.create_mask_video: Initializing OCR model...")
    print(f"mask.create_mask_video: Settings: MODEL={MODEL_NAME}, DEBUG={debug}, BATCH_SIZE={BATCH_SIZE}, LIMIT_SIDE_LEN={LIMIT_SIDE_LEN}, UNCLIP_RATIO={UNCLIP_RATIO}, MASK_DILATION={mask_dilation}, ROI={roi}")
    model_kwargs = {
        "model_name": MODEL_NAME,
        "unclip_ratio": UNCLIP_RATIO
    }
    if LIMIT_SIDE_LEN is not None:
        model_kwargs["limit_side_len"] = LIMIT_SIDE_LEN
    model = TextDetection(**model_kwargs)
    even_frame_masks = {}
    even_frame_overlays = {} if debug else None
    even_frame_full_frames = {} if debug and roi is not None else None
    q = queue.Queue(maxsize=QUEUE_MAX)
    done = object()
    def producer():
        batch_crop = []
        batch_full = [] if debug and roi is not None else None
        batch_indices = []
        for idx in range(0, total, 2):
            frame = vr[idx]
            rgb_full = frame.asnumpy()
            if roi is not None:
                x, y, w_roi, h_roi = roi
                rgb_crop = rgb_full[y:y+h_roi, x:x+w_roi]
                if batch_full is not None:
                    batch_full.append(rgb_full)
            else:
                rgb_crop = rgb_full
            batch_crop.append(rgb_crop)
            batch_indices.append(idx)
            if len(batch_crop) == BATCH_SIZE:
                q.put((np.stack(batch_crop, axis=0), 
                       np.stack(batch_full, axis=0) if batch_full else None,
                       batch_indices.copy()))
                batch_crop = []
                batch_full = [] if batch_full is not None else None
                batch_indices = []
        if batch_crop:
            q.put((np.stack(batch_crop, axis=0),
                   np.stack(batch_full, axis=0) if batch_full else None,
                   batch_indices.copy()))
        q.put(done)
    total_text_detections = 0
    def consumer():
        nonlocal total_text_detections, even_frame_masks, even_frame_overlays, even_frame_full_frames
        processed = 0
        while True:
            item = q.get()
            if item is done:
                q.task_done()
                break
            batch_crop, batch_full, batch_indices = item
            batch_bgr_crop = np.ascontiguousarray(batch_crop[..., ::-1])
            batch_bgr_full = np.ascontiguousarray(batch_full[..., ::-1]) if batch_full is not None else None
            preds = model.predict([batch_bgr_crop[i] for i in range(batch_bgr_crop.shape[0])],
                                  batch_size=batch_bgr_crop.shape[0])
            for idx_in_batch, (pred, frame_idx) in enumerate(zip(preds, batch_indices)):
                mask, dt_polys = create_single_mask(pred, h, w, roi, mask_dilation)
                even_frame_masks[frame_idx] = mask
                if debug:
                    total_text_detections += len(dt_polys)
                    if batch_bgr_full is not None:
                        frame_for_overlay = batch_bgr_full[idx_in_batch]
                        even_frame_full_frames[frame_idx] = frame_for_overlay
                    else:
                        frame_for_overlay = batch_bgr_crop[idx_in_batch]
                    overlay = create_single_overlay(frame_for_overlay, mask, dt_polys, roi)
                    even_frame_overlays[frame_idx] = overlay
            processed += len(batch_indices)
            if processed % 60 == 0 or processed >= (total // 2):
                elapsed = time.time() - start
                fps_proc = processed / elapsed
                print(f"mask.create_mask_video: Progress (even frames): {processed}/{total//2 + (total%2)} frames ({fps_proc:.2f} fps)")
            q.task_done()
    print(f"mask.create_mask_video: Phase 1 - Processing even frames through OCR (batch size: {BATCH_SIZE})...")
    ocr_start = time.time()
    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)
    t_prod.start()
    t_cons.start()
    t_prod.join()
    q.join()
    ocr_time = time.time() - ocr_start
    print(f"mask.create_mask_video: Phase 1 completed in {ocr_time:.2f}s")
    print(f"mask.create_mask_video: Phase 2 - Interpolating odd frames and writing video...")
    write_start = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(OUT_MASK, fourcc, fps, (w, h))
    overlay_writer = cv2.VideoWriter(OUT_OVERLAY, fourcc, fps, (w, h)) if debug else None
    for idx in range(total):
        if idx % 2 == 0:
            mask = even_frame_masks[idx]
            if overlay_writer is not None:
                overlay = even_frame_overlays[idx]
        else:
            left_idx = idx - 1
            right_idx = idx + 1
            if right_idx >= total:
                right_idx = left_idx
            mask_left = even_frame_masks[left_idx]
            mask_right = even_frame_masks.get(right_idx, mask_left)
            mask = cv2.bitwise_or(mask_left, mask_right)
            if overlay_writer is not None:
                if left_idx in even_frame_full_frames:
                    base_frame = even_frame_full_frames[left_idx]
                else:
                    base_frame = np.zeros((h, w, 3), np.uint8)
                overlay = base_frame.copy()
                overlay_layer = overlay.copy()
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_bool = mask_gray > 0
                overlay_layer[mask_bool] = (0, 255, 255)
                cv2.addWeighted(overlay_layer, OVERLAY_ALPHA * 0.7, overlay, 1 - OVERLAY_ALPHA * 0.7, 0, overlay)
        mask_writer.write(mask)
        if overlay_writer is not None:
            overlay_writer.write(overlay)
        if (idx + 1) % 60 == 0 or (idx + 1) >= total:
            print(f"mask.create_mask_video: Writing progress: {idx + 1}/{total} frames")
    mask_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()
    write_time = time.time() - write_start
    total_time = time.time() - start
    print(f"\nmask.create_mask_video: === Summary ===")
    print(f"mask.create_mask_video: Total frames: {total}")
    print(f"mask.create_mask_video: Even frames processed (OCR): {len(even_frame_masks)}")
    print(f"mask.create_mask_video: Odd frames interpolated (union): {total - len(even_frame_masks)}")
    print(f"mask.create_mask_video: OCR processing time: {ocr_time:.2f}s")
    print(f"mask.create_mask_video: Interpolation + writing time: {write_time:.2f}s")
    print(f"mask.create_mask_video: Total time: {total_time:.2f}s ({total/total_time:.2f} fps)")
    if debug:
        print(f"mask.create_mask_video: Total text detections (even frames only): {total_text_detections}")
        if len(even_frame_masks) > 0:
            print(f"mask.create_mask_video: Average detections per even frame: {total_text_detections/len(even_frame_masks):.2f}")
        if total_text_detections == 0:
            print(f"\nmask.create_mask_video: WARNING: NO TEXT DETECTED IN ANY FRAME!")
            print(f"mask.create_mask_video:   Possible issues:")
            print(f"mask.create_mask_video:     1. ROI is incorrect: {roi}")
            print(f"mask.create_mask_video:     2. Video has no text/captions")
            print(f"mask.create_mask_video:     3. Text is too small/blurry")
            print(f"mask.create_mask_video:   The mask and overlay videos will be blank.")
            print(f"mask.create_mask_video:   Check roi_preview.png to verify ROI placement.")
        else:
            print(f"mask.create_mask_video: Text detection successful!")
    print(f"\nmask.create_mask_video: Output files:")
    print(f"mask.create_mask_video:   Mask video: {OUT_MASK}")
    if debug:
        print(f"mask.create_mask_video:   Overlay video: {OUT_OVERLAY}")
        print(f"mask.create_mask_video:   Note: Green overlay = OCR detected, Yellow overlay = Interpolated")
    return OUT_MASK

def visualize_roi(video_path, roi, frame_idx=200, output_path="data/results/roi_preview.png"):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame = vr[frame_idx].asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x, y, w_roi, h_roi = roi
    cv2.rectangle(frame, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 3)
    cv2.putText(frame, f"ROI: {roi}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"ROI preview saved to: {output_path}")

if __name__ == "__main__":
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.stages.postprocessing.compress import compress_videos
    test_video = os.path.join("data", "examples", "34s1080_preprocessed.mp4")
    test_roi = (150, 880, 600, 200)
    visualize_roi(test_video, test_roi, frame_idx=300)
    mask_video = create_mask_video(test_video, debug=DEBUG, roi=test_roi)
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
    print("Video compression completed")
