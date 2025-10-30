# Modal Deployment Guide

## Prerequisites

1. **Install Modal CLI**
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**
   ```bash
   modal token new
   ```
   This will open a browser to authenticate.

## Setup Secrets

You need to create two secrets in Modal dashboard:

### 1. R2 Credentials (`r2-credentials`)

Go to https://modal.com/secrets and create a new secret named `r2-credentials` with:

```
R2_ACCOUNT_ID=<your_cloudflare_account_id>
R2_ACCESS_KEY_ID=<your_r2_access_key>
R2_SECRET_ACCESS_KEY=<your_r2_secret_key>
R2_BUCKET_NAME=<your_bucket_name>
```

### 2. AWS S3 Credentials (`aws-s3-credentials`)

Create another secret named `aws-s3-credentials` with:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
```

## Deploy

```bash
modal deploy modal_app.py
```

**First deployment will take ~10-15 minutes** to build the image with all dependencies.

## Usage

### Option 1: Direct Python Call

```python
import modal

# Get the function
remove_captions = modal.Function.lookup("caption-remover", "CaptionRemover.remove_captions")

# Call it
result_url = remove_captions.remote(
    video_r2_url="https://your-bucket.r2.cloudflarestorage.com/videos/input.mp4",
    model_to_use="diffueraser"  # or "propainter"
)

print(f"Result: {result_url}")
```

### Option 2: HTTP API

After deployment, Modal will give you a web endpoint URL like:
```
https://your-username--caption-remover-api-remove-captions.modal.run
```

Call it with:

```bash
curl -X POST https://your-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://your-bucket.r2.cloudflarestorage.com/videos/input.mp4",
    "model": "diffueraser"
  }'
```

Response:
```json
{
  "status": "success",
  "result_url": "https://your-bucket.r2.cloudflarestorage.com/results/input_diffueraser_result.mp4",
  "model_used": "diffueraser"
}
```

### Option 3: Command Line Testing

```bash
modal run modal_app.py --video-url "https://your-bucket.r2.cloudflarestorage.com/videos/input.mp4" --model diffueraser
```

## Performance

### First Request (Cold Start)
- Container boot: ~30s
- Model weight download (first time only): ~5-10 min
- Model loading: ~5-10s
- Inference: ~3-4 min
- **Total first request: ~15-20 min** (subsequent requests skip weight download)

### Warm Requests
- Container reuse (if within ~5 min): instant
- Model already loaded in GPU: instant
- Inference: ~3-4 min
- **Total warm request: ~3-4 min**

### Subsequent Cold Starts (after weights cached)
- Container boot: ~30s
- Model loading: ~5-10s
- Inference: ~3-4 min
- **Total: ~4-5 min**

## Monitoring

View logs and status:
```bash
modal app logs caption-remover
```

Or visit: https://modal.com/apps

## Costs Estimate (A100-80GB)

- Build time: Free
- Idle time: Free (no keep_warm)
- Active inference (~4 min): ~$0.40-0.50 per video
- Model loading overhead: ~$0.05-0.10

**Approximate: $0.50-0.60 per video processed**

## Switching to A100-40GB

Edit `modal_app.py` line 89:
```python
gpu="A100",  # Defaults to 40GB
```

This will reduce cost by ~50%.

## Troubleshooting

### Issue: Weights not downloading
Check AWS credentials are correct in the secret.

### Issue: Out of memory
- Try A100-80GB instead of 40GB
- Reduce `max_img_size` in the code (currently 720)

### Issue: Timeout
- Increase `timeout=900` to higher value (in seconds)
- For very long videos, consider splitting

### Issue: R2 upload fails
- Check R2 credentials are correct
- Verify bucket name matches

## Next Steps

1. Add ROI parameter handling (currently hardcoded)
2. Add video validation
3. Add webhook support for async processing
4. Consider batch processing for multiple videos
