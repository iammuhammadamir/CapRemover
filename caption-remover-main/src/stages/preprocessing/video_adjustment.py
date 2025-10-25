"""Video adjustment scaffolding."""

from __future__ import annotations

from typing import Any

import math
import subprocess
import os


def adjust_resolution_and_fps(
    *,
    video_path: str,
    max_resolution: int,
    target_fps: float,
    precheck_info: Any,
    output_path: str,
) -> None:
    """Normalize video resolution and frame rate before mask creation."""

    if precheck_info.is_compliant:
        return

    filters = []
    if precheck_info.needs_fps_fix:
        filters.append(f"fps={target_fps}")
    if precheck_info.needs_resize:
        scale = max_resolution / max(precheck_info.width, precheck_info.height)
        w = max(2, int(math.floor(precheck_info.width * scale / 2) * 2))
        h = max(2, int(math.floor(precheck_info.height * scale / 2) * 2))
        filters.append(f"scale={w}:{h}")

    if not filters:
        return

    # Use H.265 encoding for better compression
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def downscale_video(
    video_path: str,
    max_dimension: int,
    output_path: str = None,
) -> str:
    """Downscale a video to a maximum dimension while maintaining aspect ratio.
    
    Args:
        video_path: Path to input video
        max_dimension: Maximum width or height (whichever is larger)
        output_path: Optional output path. If None, adds _downscaled suffix
    
    Returns:
        Path to the downscaled video
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_downscaled{ext}"
    
    # Use ffmpeg scale filter with max dimension
    # The scale filter will maintain aspect ratio automatically
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"scale='if(gt(iw,ih),min({max_dimension},iw),-2)':'if(gt(iw,ih),-2,min({max_dimension},ih))'",
        "-c:v",
        "libx265",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def crop_video_to_roi(
    video_path: str,
    roi: tuple[int, int, int, int],
    expansion_factor: float = 2.0,
    output_path: str = None,
) -> tuple[str, tuple[int, int, int, int]]:
    """Crop a video to a region around the ROI.
    
    Args:
        video_path: Path to input video
        roi: (x, y, w, h) region of interest
        expansion_factor: Factor to expand ROI by (e.g., 2.0 for 2x size)
        output_path: Optional output path. If None, adds _cropped suffix
    
    Returns:
        Tuple of (cropped_video_path, new_roi_in_cropped_coords)
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_cropped{ext}"
    
    # Get video dimensions
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    vid_w, vid_h = map(int, result.stdout.strip().split('x'))
    
    # Calculate expanded ROI
    x, y, w, h = roi
    center_x = x + w // 2
    center_y = y + h // 2
    
    new_w = int(w * expansion_factor)
    new_h = int(h * expansion_factor)
    
    # Calculate crop coordinates (ensure even numbers for video encoding)
    crop_x = max(0, center_x - new_w // 2)
    crop_y = max(0, center_y - new_h // 2)
    crop_w = min(new_w, vid_w - crop_x)
    crop_h = min(new_h, vid_h - crop_y)
    
    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    crop_x = crop_x - (crop_x % 2)
    crop_y = crop_y - (crop_y % 2)
    
    # Calculate new ROI coordinates in the cropped video
    new_roi_x = x - crop_x
    new_roi_y = y - crop_y
    new_roi = (new_roi_x, new_roi_y, w, h)
    
    print(f"Cropping video from {vid_w}x{vid_h} to {crop_w}x{crop_h}")
    print(f"Original ROI: ({x}, {y}, {w}, {h})")
    print(f"Crop region: ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
    print(f"New ROI in cropped coords: {new_roi}")
    
    # Crop using ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-c:v",
        "libx265",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_path, new_roi
