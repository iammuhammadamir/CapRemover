# ROI API Documentation

## Overview

The Caption Remover API now supports **normalized ROI coordinates** (0.0 to 1.0 range), making it resolution-independent and easy to use across different video dimensions.

---

## API Request Format

### Endpoint
```
POST https://mramir--caption-remover-web.modal.run
```

### Request Body

```json
{
  "video_url": "https://your-r2-url/video.mp4",
  "model": "diffueraser",
  "roi": {
    "x": 0.212,
    "y": 0.677,
    "width": 0.564,
    "height": 0.094
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video_url` | string | ✅ Yes | R2 URL of the video to process |
| `model` | string | No | Model to use: `"diffueraser"` (default) or `"propainter"` |
| `roi` | object | No | Normalized ROI coordinates (see below) |

### ROI Object

| Field | Type | Required | Range | Description |
|-------|------|----------|-------|-------------|
| `x` | float | ✅ Yes | 0.0 - 1.0 | Horizontal position (0.0 = left edge, 1.0 = right edge) |
| `y` | float | ✅ Yes | 0.0 - 1.0 | Vertical position (0.0 = top edge, 1.0 = bottom edge) |
| `width` | float | ✅ Yes | 0.0 - 1.0 | Width as proportion of video width |
| `height` | float | ✅ Yes | 0.0 - 1.0 | Height as proportion of video height |

**Note:** If `roi` is not provided, the API uses a default ROI optimized for bottom captions.

---

## Response Format

### Success Response (200)

```json
{
  "status": "success",
  "result_url": "https://your-r2-url/results/video_diffueraser_result.mp4",
  "model_used": "diffueraser"
}
```

### Error Responses

#### Missing video_url (400)
```json
{
  "error": "video_url is required"
}
```

#### Invalid model (400)
```json
{
  "error": "model must be 'propainter' or 'diffueraser'"
}
```

#### Invalid ROI format (400)
```json
{
  "error": "roi must contain x, y, width, height"
}
```

#### Invalid ROI values (400)
```json
{
  "error": "roi values must be between 0 and 1"
}
```

---

## Examples

### Example 1: Using Default ROI

```python
import requests

response = requests.post(
    "https://mramir--caption-remover-web.modal.run",
    json={
        "video_url": "https://viewmax-assets.r2.cloudflarestorage.com/video.mp4",
        "model": "diffueraser"
    }
)

result = response.json()
print(result["result_url"])
```

### Example 2: Custom ROI for Top Captions

```python
import requests

# ROI for captions at top of video (top 15%)
response = requests.post(
    "https://mramir--caption-remover-web.modal.run",
    json={
        "video_url": "https://viewmax-assets.r2.cloudflarestorage.com/video.mp4",
        "model": "diffueraser",
        "roi": {
            "x": 0.1,      # 10% from left
            "y": 0.05,     # 5% from top
            "width": 0.8,  # 80% of video width
            "height": 0.15 # 15% of video height
        }
    }
)

result = response.json()
print(result["result_url"])
```

### Example 3: Custom ROI for Center Captions

```python
import requests

# ROI for captions in center of video
response = requests.post(
    "https://mramir--caption-remover-web.modal.run",
    json={
        "video_url": "https://viewmax-assets.r2.cloudflarestorage.com/video.mp4",
        "model": "propainter",
        "roi": {
            "x": 0.25,     # 25% from left
            "y": 0.4,      # 40% from top
            "width": 0.5,  # 50% of video width
            "height": 0.2  # 20% of video height
        }
    }
)

result = response.json()
print(result["result_url"])
```

---

## How It Works

### 1. Video Preprocessing
The video is first preprocessed to a maximum resolution of 1600px (maintaining aspect ratio) and normalized to 24 FPS.

### 2. ROI Conversion
The normalized ROI coordinates are converted to pixel coordinates based on the **preprocessed** video dimensions:

```python
# Example: 900x1600 preprocessed video
roi_x = int(0.212 * 900) = 190 → 190 (even)
roi_y = int(0.677 * 1600) = 1083 → 1082 (even)
roi_w = int(0.564 * 900) = 507 → 506 (even)
roi_h = int(0.094 * 1600) = 150 → 150 (even)

# Result: (190, 1082, 506, 150)
```

**Note:** Coordinates are adjusted to even numbers as required by video encoding.

### 3. Processing Pipeline
1. **Mask Creation**: OCR detects text within the ROI
2. **Cropping**: Video is cropped to 2x ROI size for context
3. **Inpainting**: ProPainter + DiffuEraser remove captions
4. **Compositing**: Result is composited back to full video

---

## Default ROI

If no ROI is provided, the API uses:

```json
{
  "x": 0.212,
  "y": 0.677,
  "width": 0.564,
  "height": 0.094
}
```

This default is optimized for **bottom-center captions** (typical for social media videos).

**Visualization:**
```
┌─────────────────────────┐
│                         │
│                         │
│      Video Content      │
│                         │
│                         │
│    ┌─────────────┐     │ ← 67.7% from top
│    │   Caption   │     │
│    └─────────────┘     │ ← 9.4% height
└─────────────────────────┘
     ↑           ↑
   21.2%      56.4% width
```

---

## Frontend Integration

### JavaScript Example

```javascript
// User draws ROI on video preview
const videoElement = document.getElementById('video');
const canvas = document.getElementById('canvas');

// Get video dimensions
const videoWidth = videoElement.videoWidth;
const videoHeight = videoElement.videoHeight;

// User's selection in pixels (from canvas drawing)
const selection = {
  x: 200,
  y: 800,
  width: 500,
  height: 150
};

// Convert to normalized coordinates
const normalizedROI = {
  x: selection.x / videoWidth,
  y: selection.y / videoHeight,
  width: selection.width / videoWidth,
  height: selection.height / videoHeight
};

// Send to API
const response = await fetch('https://mramir--caption-remover-web.modal.run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    video_url: videoUrl,
    model: 'diffueraser',
    roi: normalizedROI
  })
});

const result = await response.json();
console.log('Result:', result.result_url);
```

---

## Tips for Choosing ROI

### 1. **Include Padding**
Add 10-20% padding around text to ensure complete removal:
```json
{
  "x": 0.15,      // Start 5% before text
  "y": 0.65,      // Start 5% above text
  "width": 0.7,   // Extend 5% beyond text
  "height": 0.15  // Extend 5% below text
}
```

### 2. **Avoid Edges**
Keep ROI at least 2-3% away from video edges to prevent artifacts.

### 3. **Test Different Sizes**
- **Smaller ROI**: Faster processing, but may miss text
- **Larger ROI**: Slower processing, but more thorough

### 4. **Common Presets**

**Bottom Captions (TikTok/Instagram):**
```json
{"x": 0.1, "y": 0.7, "width": 0.8, "height": 0.15}
```

**Top Captions (YouTube):**
```json
{"x": 0.1, "y": 0.05, "width": 0.8, "height": 0.15}
```

**Center Captions:**
```json
{"x": 0.2, "y": 0.4, "width": 0.6, "height": 0.2}
```

**Full Width Bottom:**
```json
{"x": 0.0, "y": 0.75, "width": 1.0, "height": 0.25}
```

---

## Troubleshooting

### Issue: Text Not Fully Removed

**Solution:** Increase ROI size by 10-20%:
```json
{
  "x": 0.15,      // Was 0.2
  "y": 0.6,       // Was 0.65
  "width": 0.7,   // Was 0.6
  "height": 0.2   // Was 0.15
}
```

### Issue: Processing Too Slow

**Solution:** Reduce ROI size to minimum needed:
```json
{
  "x": 0.25,      // Tighter bounds
  "y": 0.7,
  "width": 0.5,   // Smaller width
  "height": 0.1   // Smaller height
}
```

### Issue: Artifacts at ROI Edges

**Solution:** Ensure ROI has padding and doesn't touch video edges:
```json
{
  "x": 0.05,      // At least 5% from edge
  "y": 0.7,
  "width": 0.9,   // Leave 5% on right
  "height": 0.25  // Leave 5% on bottom
}
```

---

## Migration Guide

### Old API (Hardcoded ROI)

```python
# Old: No ROI parameter
response = requests.post(API_URL, json={
    "video_url": video_url,
    "model": "diffueraser"
})
```

### New API (Normalized ROI)

```python
# New: Optional ROI parameter
response = requests.post(API_URL, json={
    "video_url": video_url,
    "model": "diffueraser",
    "roi": {
        "x": 0.212,
        "y": 0.677,
        "width": 0.564,
        "height": 0.094
    }
})
```

**Backward Compatible:** Old requests without ROI still work with default values!

---

## Performance Notes

- **Default ROI**: ~9-15 minutes for 1339 frames
- **Smaller ROI**: Can reduce processing time by 20-30%
- **Larger ROI**: May increase processing time by 20-40%

**Timeout:** 60 minutes (3600 seconds)

---

## Support

For issues or questions, contact the development team or check the logs in the Modal dashboard.
