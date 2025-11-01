# Cost Optimization - Single ASGI App Pattern

## Problem Identified

### Previous Implementation (Expensive ❌)
```
web endpoint      → 1 container per request
status endpoint   → 1 container per poll (50+ containers!)
CaptionRemover    → 2 GPU containers

Total: 50+ CPU containers + 2 GPU containers = EXPENSIVE!
```

**Cost breakdown:**
- Each `@modal.web_endpoint` spawns a new container
- 50 status polls = 50 separate containers
- Each container has startup overhead
- Charged per container-second

### Root Cause
Using multiple `@modal.web_endpoint` decorators creates separate functions, each spawning its own container on every call.

---

## Solution: Single FastAPI ASGI App ✅

### New Implementation (Efficient)
```
FastAPI ASGI app  → 1 container handles ALL requests
  ├─ POST /submit     (job submission)
  └─ GET /status/{id} (status checks - 50+ polls, same container!)

CaptionRemover    → 1 GPU container per job

Total: 1 CPU container + 1 GPU container = MUCH CHEAPER!
```

**Cost savings:**
- ✅ Single container handles all HTTP requests
- ✅ No cold starts for status checks
- ✅ Container stays warm between polls
- ✅ Only pay for GPU processing time

---

## Implementation Changes

### modal_app.py

#### Before (Multiple Endpoints):
```python
@app.function()
@modal.web_endpoint(method="POST")
def web(data: dict):
    # Each call = new container
    ...

@app.function()
@modal.web_endpoint(method="GET")
def status(job_id: str):
    # Each call = new container (50+ containers!)
    ...
```

#### After (Single ASGI App):
```python
from fastapi import FastAPI

web_app = FastAPI()

@web_app.post("/submit")
async def submit_job(request: Request):
    # Handled by single container
    ...

@web_app.get("/status/{job_id}")
async def check_status(job_id: str):
    # Same container handles all status checks!
    ...

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app  # Single container serves all routes
```

### deployment_test.py

#### Updated URLs:
```python
# Before
MODAL_API_URL = "https://caption-remover--caption-remover-web.modal.run/"
status_url = MODAL_API_URL.replace('-web', '-status')

# After
MODAL_API_URL = "https://caption-remover--caption-remover-fastapi-app.modal.run"
submit_url = f"{MODAL_API_URL}/submit"
status_url = f"{MODAL_API_URL}/status/{job_id}"
```

---

## API Endpoints

### 1. Submit Job
**URL:** `POST https://caption-remover--caption-remover-fastapi-app.modal.run/submit`

**Request:**
```json
{
  "video_url": "https://...",
  "model": "diffueraser",
  "roi": {...},
  "webhook_url": "https://..."
}
```

**Response:**
```json
{
  "status": "processing",
  "job_id": "fc-01HZXY...",
  "message": "Job submitted successfully..."
}
```

### 2. Check Status
**URL:** `GET https://caption-remover--caption-remover-fastapi-app.modal.run/status/{job_id}`

**Response (Processing - 202):**
```json
{
  "status": "processing",
  "job_id": "fc-01HZXY...",
  "message": "Job is still processing..."
}
```

**Response (Complete - 200):**
```json
{
  "status": "completed",
  "job_id": "fc-01HZXY...",
  "result_url": "https://..."
}
```

---

## Cost Comparison

### Scenario: 1 video processing job with 50 status polls

#### Before (Multiple web_endpoints):
```
Container Usage:
- web endpoint:    1 call  × 1s    = 1 container-second
- status endpoint: 50 calls × 1s   = 50 container-seconds
- GPU processing:  1 call  × 540s  = 540 GPU-seconds

Total CPU: 51 container-seconds
Total GPU: 540 GPU-seconds
```

#### After (Single ASGI app):
```
Container Usage:
- FastAPI app:     51 calls × 0.1s = 5.1 container-seconds (warm)
- GPU processing:  1 call  × 540s  = 540 GPU-seconds

Total CPU: 5.1 container-seconds (90% reduction!)
Total GPU: 540 GPU-seconds (same)
```

### Estimated Savings
- **CPU costs:** ~90% reduction
- **Cold starts:** Eliminated for status checks
- **Response time:** Faster (no container spin-up)

---

## Benefits

✅ **90% reduction in CPU container usage**  
✅ **Single container handles all HTTP requests**  
✅ **No cold starts for status checks**  
✅ **Faster response times**  
✅ **Follows Modal's recommended pattern**  
✅ **Easier to maintain (one FastAPI app)**  
✅ **Better for production scaling**  

---

## Deployment

### 1. Deploy Updated Code
```bash
python3 -m modal deploy modal_app.py
```

**Expected output:**
```
✓ Created objects.
├── 🔨 Created mount /root/src
├── 🔨 Created mount /root/main.py
├── 🔨 Created CaptionRemover => caption-remover-CaptionRemover
└── 🔨 Created fastapi_app => caption-remover-fastapi-app
✓ App deployed! 🎉

View Deployment: https://modal.com/apps/...
```

### 2. Update API URL
The new endpoint will be:
```
https://caption-remover--caption-remover-fastapi-app.modal.run
```

### 3. Test
```bash
python deployment_test.py
```

---

## Monitoring

### Check Dashboard
After deployment, you should see:
- ✅ **fastapi_app** (CPU) - 1 container, multiple calls
- ✅ **CaptionRemover** (GPU A100) - 1 container per job

### What to Look For
- **fastapi_app calls:** Should show 51+ calls (1 submit + 50 status checks)
- **fastapi_app containers:** Should show 1 container (not 51!)
- **CaptionRemover calls:** Should show 1 call per video
- **CaptionRemover containers:** Should show 1 container per job

---

## Troubleshooting

### Issue: 404 on /submit or /status

**Cause:** Old deployment still active

**Solution:**
```bash
# Stop old deployment
modal app stop caption-remover

# Redeploy
python3 -m modal deploy modal_app.py
```

### Issue: Still seeing multiple containers

**Cause:** Using old web_endpoint URLs

**Solution:** Make sure `deployment_test.py` uses:
```python
MODAL_API_URL = "https://caption-remover--caption-remover-fastapi-app.modal.run"
```

### Issue: Status checks timing out

**Cause:** Container cold start

**Solution:** First status check might be slow, subsequent checks will be fast (container stays warm)

---

## Best Practices

### 1. Keep Container Warm
For production, consider:
```python
@app.function(
    keep_warm=1  # Keep 1 container always warm
)
@modal.asgi_app()
def fastapi_app():
    return web_app
```

### 2. Add Health Check
```python
@web_app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 3. Add Metrics
```python
@web_app.get("/metrics")
async def metrics():
    return {
        "active_jobs": len(active_jobs),
        "completed_jobs": completed_count
    }
```

---

## References

- [Modal Request Timeouts](https://modal.com/docs/guide/webhooks#request-timeouts)
- [Modal ASGI Apps](https://modal.com/docs/guide/webhooks#asgi-apps)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Document OCR Example](https://modal.com/docs/examples/doc_ocr_webapp)

---

## Summary

**Before:** 50+ containers for a single job = expensive  
**After:** 1 container handles all requests = 90% cost reduction  

This is the **recommended Modal pattern** for long-running jobs with polling.
