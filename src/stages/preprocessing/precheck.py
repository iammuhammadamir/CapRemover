"""Precheck scaffolding for preprocessing stage."""

from __future__ import annotations

from types import SimpleNamespace

import decord


def run_precheck(*, video_path: str, max_resolution: int, target_fps: float) -> SimpleNamespace:
    """Inspect the input video and report compliance."""

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps() or 0.0)
    frame = vr[0]
    height, width = int(frame.shape[0]), int(frame.shape[1])
    needs_fps = fps > target_fps
    needs_resize = max(width, height) > max_resolution
    return SimpleNamespace(
        is_compliant=not (needs_fps or needs_resize),
        fps=fps,
        width=width,
        height=height,
        needs_fps_fix=needs_fps,
        needs_resize=needs_resize,
    )

