# IMPORTANT: Import torch FIRST to initialize CUDA before PaddlePaddle
# This prevents CUDA initialization conflicts between PyTorch and PaddlePaddle
import torch
torch.cuda.init()  # Force CUDA initialization

import time
import sys
from src.stages.create_mask import create_mask_video, visualize_roi
from src.stages.preprocessing import preprocess_video, run_precheck, downscale_video, crop_video_to_roi
from src.stages.inpaint.inpaint import run_inpainting
from src.stages.postprocessing.composite import composite_inpainted_region
from src.stages.postprocessing.compress import compress_videos


class TeeOutput:
    """Write to both console and file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', buffering=1)  # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main():
    total_start = time.time()
    
    # Configuration for long.mp4
    video = "data/examples/long.mp4"
    max_resolution = 1600  # 900p max dimension (900p = 1600x900)
    target_fps = 24.0
    debug = True  # Set to True to enable video compression

    # TODO: Adjust ROI for long.mp4 - use roi.py to find the correct values
    # Run: python roi.py to find the correct x, y,`` width, height for the caption region
    roi = (191, 1083, 508, 150)  # SCALED ROI for preprocessed video (900x1600)
    # for long.mp4 use: (230,1300,610,180)
    
    # Optimization settings
    raft_iter = 12  # Reduce RAFT iterations 
    enable_pre_inference = False  # Skip DiffuEraser keyframe pass for static captions



    # [1/5] Preprocessing
    print("=" * 60)
    print("[1/5] PREPROCESSING VIDEO")
    print("=" * 60)
    preprocess_start = time.time()
    processed_video = preprocess_video(video_path=video, max_resolution=max_resolution, target_fps=target_fps)
    preprocess_time = time.time() - preprocess_start

    info = run_precheck(video_path=processed_video, max_resolution=max_resolution, target_fps=target_fps)
    print(f"\nProcessed video: {processed_video}")
    print(f"Resolution: {info.width}x{info.height}, FPS: {info.fps}")
    print(f"✓ Preprocessing completed in {preprocess_time:.2f}s")

    # [2/5] Create mask
    print("\n" + "=" * 60)
    print("[2/5] CREATING MASK")
    print("=" * 60)
    visualize_roi(processed_video, roi, frame_idx=50)
    
    mask_start = time.time()
    mask_video = create_mask_video(video_path=processed_video, debug=debug, roi=roi)
    mask_time = time.time() - mask_start
    print(f"\nMask video: {mask_video}")
    print(f"✓ Mask creation completed in {mask_time:.2f}s")
    
    # [3/5] Crop to 2x ROI region for inpainting
    print("\n" + "=" * 60)
    print("[3/5] CROPPING TO 2X ROI REGION")
    print("=" * 60)
    crop_start = time.time()
    cropped_video, new_roi = crop_video_to_roi(
        processed_video, 
        roi, 
        expansion_factor=2.0,
        output_path="data/results/video_cropped_2x_roi.mp4"
    )
    cropped_mask, _ = crop_video_to_roi(
        mask_video, 
        roi, 
        expansion_factor=2.0,
        output_path="data/results/mask_cropped_2x_roi.mp4"
    )
    crop_time = time.time() - crop_start
    print(f"\nCropped video: {cropped_video}")
    print(f"Cropped mask: {cropped_mask}")
    print(f"✓ Cropping completed in {crop_time:.2f}s")
    
    # [4/5] Run inpainting on cropped region
    print("\n" + "=" * 60)
    print("[4/5] INPAINTING (2X ROI REGION)")
    print("=" * 60)
    inpaint_start = time.time()
    propainter_output, diffueraser_output = run_inpainting(
        video_path=cropped_video, 
        mask_path=cropped_mask, 
        mask_dilation=8, 
        max_img_size=720,
        raft_iter=raft_iter,
        enable_pre_inference=enable_pre_inference
    )
    inpaint_time = time.time() - inpaint_start
    
    # [5/6] Composite inpainted regions back onto preprocessed video
    print("\n" + "=" * 60)
    print("[5/6] COMPOSITING RESULTS")
    print("=" * 60)
    composite_start = time.time()
    propainter_composited, diffueraser_composited = composite_inpainted_region(
        preprocessed_video=processed_video,
        roi=roi,
        propainter_result=propainter_output,
        diffueraser_result=diffueraser_output
    )
    composite_time = time.time() - composite_start
    print(f"✓ Compositing completed in {composite_time:.2f}s")
    
    # [6/6] Compress videos (if debug enabled)
    if debug:
        print("\n" + "=" * 60)
        print("[6/6] COMPRESSING VIDEOS")
        print("=" * 60)
        compress_start = time.time()
        overlay_video = "data/results/overlay_video.mp4"
        # Compress all intermediate and final videos
        compress_videos(propainter_output, diffueraser_output, mask_video, overlay_video=overlay_video, debug=debug)
        compress_time = time.time() - compress_start
        print(f"✓ Video compression completed in {compress_time:.2f}s")
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"  [1] Preprocessing:  {preprocess_time:6.2f}s")
    print(f"  [2] Mask creation:  {mask_time:6.2f}s")
    print(f"  [3] ROI Cropping:   {crop_time:6.2f}s")
    print(f"  [4] Inpainting:     {inpaint_time:6.2f}s")
    print(f"  [5] Compositing:    {composite_time:6.2f}s")
    if debug:
        print(f"  [6] Compression:    {compress_time:6.2f}s")
    print(f"  " + "-" * 35)
    print(f"  TOTAL TIME:        {total_time:6.2f}s ({total_time/60:.1f} min)")
    print(f"\n📁 INTERMEDIATE FILES:")
    print(f"  • Cropped video:   {cropped_video}")
    print(f"  • Cropped mask:    {cropped_mask}")
    print(f"  • Propainter:      {propainter_output}")
    print(f"  • DiffuEraser:     {diffueraser_output}")
    print(f"\n📁 FINAL COMPOSITED VIDEOS:")
    print(f"  • Propainter:      {propainter_composited}")
    print(f"  • DiffuEraser:     {diffueraser_composited}")
    print(f"\n✨ DiffuEraser composited is the refined final result!")

if __name__ == "__main__":
    # Redirect stdout to both terminal and file
    tee = TeeOutput("terminal_output.txt")
    sys.stdout = tee
    
    try:
        main()
    finally:
        # Restore original stdout and close log file
        sys.stdout = tee.terminal
        tee.close()

