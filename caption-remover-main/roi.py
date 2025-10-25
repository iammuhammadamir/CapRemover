#!/usr/bin/env python3

import sys
from src.stages.create_mask import visualize_roi

video = "data/examples/snail_preprocessed.mp4"
roi = (145, 700, 600, 180)  # (x, y, width, height)
frame_idx = 300

visualize_roi(video, roi, frame_idx)