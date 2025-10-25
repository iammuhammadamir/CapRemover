"""Orchestrates preprocessing steps."""

from __future__ import annotations

import os

from .precheck import run_precheck
from .video_adjustment import adjust_resolution_and_fps


def preprocess_video(
    *,
    video_path: str,
    max_resolution: int,
    target_fps: float,
    output_path: str | None = None,
) -> str:
    """Run preprocessing pipeline for a video input."""

    info = run_precheck(
        video_path=video_path,
        max_resolution=max_resolution,
        target_fps=target_fps,
    )
    if info.is_compliant:
        return video_path

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_preprocessed{ext or '.mp4'}"

    adjust_resolution_and_fps(
        video_path=video_path,
        max_resolution=max_resolution,
        target_fps=target_fps,
        precheck_info=info,
        output_path=output_path,
    )
    return output_path

