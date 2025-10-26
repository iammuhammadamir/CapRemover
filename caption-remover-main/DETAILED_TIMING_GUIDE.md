# Detailed Timing Instrumentation Guide

## Overview
Added granular timing measurements to the inpainting pipeline to identify optimization opportunities.

---

## ProPainter Timing Breakdown

**File:** `src/stages/inpaint/DiffuEraser/propainter/inference.py`

### Timing Labels:
```
[3a] I/O (read video/mask, preprocess)
     - Video frame loading
     - Mask reading and preprocessing
     - Tensor conversion and device transfer

[3b] RAFT optical flow (X iters)
     - Optical flow computation using RAFT model
     - Most computationally expensive part of ProPainter
     - Time scales with iteration count (current: 12)

[3c] Flow completion
     - Bidirectional flow prediction
     - Flow field inpainting
     - Combines ground truth and predicted flows

[3d] Video write (X frames)
     - Encoding final ProPainter output to MP4
     - Frame compositing

ProPainter total: Sum of all above steps
```

---

## DiffuEraser Timing Breakdown

**File:** `src/stages/inpaint/DiffuEraser/diffueraser/diffueraser.py`

### Timing Labels:
```
[4a] I/O (read video/mask/priori)
     - Reading ProPainter output (priori)
     - Loading masks and validation images

[4b] Image preprocessing
     - PIL image resizing and format conversion
     - Tensor preparation for VAE

[4c] VAE encode (X frames, batch=Y)
     - Encoding frames to latent space
     - Shows number of frames and batch size
     - Potential torch.compile optimization target

[4d] Deep copy masks/frames
     - Creating backup copies for compositing
     - May be optimization opportunity (reduce copies)

[4e] Pre-inference: Keyframe pass (if enabled)
     └── [4e.1] Keyframe sampling (X frames)
         - Selecting representative keyframes

     └── [4e.2] Add noise
         - Noise scheduling for diffusion

     └── [4e.3] UNet+BrushNet inference (X steps)
         - Main diffusion model forward pass
         - MAJOR BOTTLENECK for torch.compile
         - Number of inference steps affects time

     └── [4e.4] VAE decode (X frames)
         - Decoding latents back to pixel space

     └── [4e.5] Replace frames in buffer
         - Updating frame buffer with refined keyframes

[4f] Main inference: Full video pass
     └── [4f.1] Add noise to X frames
         - Noise preparation for all frames

     └── [4f.2] UNet+BrushNet+VAE pipeline (X steps, Y frames)
         - End-to-end diffusion pipeline
         - LARGEST BOTTLENECK (~150-200s for 1320 frames)
         - Dynamic shape issue prevents torch.compile optimization

[4g] Compositing: Blending inpainted frames
     └── [4g.1] Mask blurring (X frames)
         - Gaussian blur for smooth transitions

     └── [4g.2] Frame blending (X frames)
         - Alpha compositing inpainted and original frames

     └── [4g.3] Video encoding (X frames)
         - Writing final DiffuEraser output

DiffuEraser total: Sum of all above steps
```

---

## How to Use This Information

### 1. Identify Bottlenecks
Run the pipeline and look for the largest time values:
```bash
python main.py 2>&1 | grep -E "\[[0-9][a-z]\]"
```

### 2. Example Output Interpretation
```
[3b] RAFT optical flow (12 iters): 95.32s  ← BOTTLENECK
[3c] Flow completion: 12.45s
[4f.2] UNet+BrushNet+VAE pipeline (2 steps, 1320 frames): 187.54s  ← MAJOR BOTTLENECK
```

### 3. Optimization Priorities (Based on Timing)
1. **[4f.2] UNet+BrushNet+VAE pipeline** (~180-200s)
   - Requires static padding for torch.compile
   - Consider reducing inference steps (2 → 1?)
   - Test mixed precision (FP16/BFloat16)

2. **[3b] RAFT optical flow** (~90-120s)
   - Reduce iterations: 12 → 8 → 6 (trade-off: accuracy vs speed)
   - Potential torch.compile target (likely more stable shapes)

3. **[4c] VAE encode** (~20-40s)
   - Good candidate for torch.compile (fixed batch size of 4)
   - Already uses batching efficiently

4. **[4e.3] / [4f.2] UNet inference** (combined ~150-200s)
   - Dynamic shapes prevent direct optimization
   - Would need significant refactoring (padding strategy)

---

## Optimization Strategies by Timing Section

### Fast Wins (< 1 day):
- **[3b] RAFT iterations**: Test 12 → 10 → 8 → 6 iters
- **[4g.3] Video encoding**: Already optimized (NVENC)
- **[4b] Preprocessing**: Minimal time, skip

### Medium Effort (2-3 days):
- **[4c] VAE encode**: Apply torch.compile (fixed shapes, should work)
- **[3b] RAFT model**: Apply torch.compile to RAFT
- **Test fewer inference steps**: 2-step → 1-step PCM

### High Effort (1-2 weeks):
- **[4f.2] UNet pipeline**: Implement static padding + masking
  - Pad all videos to 2048 frames
  - Track real vs padded regions
  - torch.compile on fixed shape
  - Expected savings: 60-80s per video

---

## Performance Tracking Template

Copy this and fill in after each optimization:

```
BASELINE:
- [3b] RAFT:        ___s
- [3c] Flow:        ___s
- [4c] VAE encode:  ___s
- [4f.2] UNet:      ___s
- Total Inpainting: ___s

AFTER OPTIMIZATION X:
- [3b] RAFT:        ___s (Δ: ___)
- [3c] Flow:        ___s (Δ: ___)
- [4c] VAE encode:  ___s (Δ: ___)
- [4f.2] UNet:      ___s (Δ: ___)
- Total Inpainting: ___s (Δ: ___)
```

---

## Notes

- All times are in seconds
- Frame counts shown in parentheses for context
- Times will vary based on:
  - Video resolution
  - Number of frames
  - GPU model (H100 vs A100 vs RTX)
  - GPU utilization (multi-tenancy)

---

**Last Updated:** $(date +%Y-%m-%d)
