import subprocess
import os


def composite_inpainted_region(
    preprocessed_video: str,
    roi: tuple[int, int, int, int],
    propainter_result: str,
    diffueraser_result: str,
    output_dir: str = "data/results",
    expansion_factor: float = 2.0
) -> tuple[str, str]:
    """
    Composite the inpainted results back onto the preprocessed video.
    
    Args:
        preprocessed_video: Path to the preprocessed video
        roi: (x, y, w, h) region of interest that was used for cropping
        propainter_result: Path to propainter output
        diffueraser_result: Path to diffueraser output
        output_dir: Directory for output videos
        expansion_factor: Factor used to expand ROI (should match crop_video_to_roi)
    
    Returns:
        Tuple of (propainter_composited_path, diffueraser_composited_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get preprocessed video dimensions
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        preprocessed_video
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    vid_w, vid_h = map(int, result.stdout.strip().split('x'))
    
    # Calculate crop position (same logic as crop_video_to_roi)
    x, y, w, h = roi
    center_x = x + w // 2
    center_y = y + h // 2
    
    new_w = int(w * expansion_factor)
    new_h = int(h * expansion_factor)
    
    crop_x = max(0, center_x - new_w // 2)
    crop_y = max(0, center_y - new_h // 2)
    crop_w = min(new_w, vid_w - crop_x)
    crop_h = min(new_h, vid_h - crop_y)
    
    # Make even
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    crop_x = crop_x - (crop_x % 2)
    crop_y = crop_y - (crop_y % 2)
    
    print(f"\n=== Compositing Settings ===")
    print(f"Preprocessed video: {vid_w}x{vid_h}")
    print(f"Crop region used: {crop_w}x{crop_h} at ({crop_x}, {crop_y})")
    print(f"Overlay position: ({crop_x}, {crop_y})")
    
    # Output paths
    propainter_composited = os.path.join(output_dir, "propainter_composited.mp4")
    diffueraser_composited = os.path.join(output_dir, "diffueraser_composited.mp4")
    
    # Composite Propainter
    print(f"\nCompositing Propainter result...")
    _composite_single(
        preprocessed_video,
        propainter_result,
        propainter_composited,
        crop_w, crop_h,
        crop_x, crop_y
    )
    print(f"✓ Propainter composited: {propainter_composited}")
    
    # Composite DiffuEraser
    print(f"\nCompositing DiffuEraser result...")
    _composite_single(
        preprocessed_video,
        diffueraser_result,
        diffueraser_composited,
        crop_w, crop_h,
        crop_x, crop_y
    )
    print(f"✓ DiffuEraser composited: {diffueraser_composited}")
    
    return propainter_composited, diffueraser_composited


def _composite_single(
    base_video: str,
    overlay_video: str,
    output_path: str,
    target_w: int,
    target_h: int,
    overlay_x: int,
    overlay_y: int
):
    """
    Composite a single inpainted result onto the base video.
    
    Scales the overlay to target dimensions, then overlays at the specified position.
    """
    cmd = [
        "ffmpeg",
        "-i", base_video,
        "-i", overlay_video,
        "-filter_complex",
        f"[1:v]scale={target_w}:{target_h}[scaled];[0:v][scaled]overlay={overlay_x}:{overlay_y}",
        "-c:a", "copy",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


