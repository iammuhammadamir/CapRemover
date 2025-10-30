# Modal Documentation Quick Reference

## Core Concepts

### **cold_start_performance_modal_docs.md**
- **Cold starts**: Time when new containers boot (vs reusing warm containers)
- **Optimization strategies**: 
  - Reduce queueing: `scaledown_window` (60s-20min), `min_containers`, `buffer_containers`
  - Reduce initialization: Move work to `@enter`, use memory snapshots, concurrent IO for model loading

### **modal_app_modal_docs.md**
- **App**: Container for functions/classes deployed together
- **Key methods**: `.run()` (ephemeral), `.deploy()` (persistent), `.function()`, `.cls()`
- **Configuration**: Can set default image, secrets, volumes for all functions in app

### **modal_cls_modal_docs.md**
- **@app.cls()**: Decorator for classes with lifecycle hooks (`@enter`, `@exit`)
- **Methods pooling**: Multiple containers can run class methods concurrently
- **`.with_options()`**: Override GPU, memory, timeout at runtime without redeploying

### **modal_enter_modal_docs.md**
- **@modal.enter()**: Runs once when container starts
- **Use case**: Load models, establish connections - containers not marked "warm" until enter completes
- **Cold start strategy**: Moves initialization latency to warm-up period

### **modal_exit_modal_docs.md**
- **@modal.exit()**: Runs when container exits
- **Use case**: Cleanup, close connections, save state

### **modal_method_modal_docs.md**
- **@modal.method()**: Transform class methods into Modal Functions
- **Usage**: Must be inside `@app.cls()` decorated class

## Storage & Secrets

### **modal_volume_modal_docs.md**
- **Volume**: Persistent file storage shared across containers
- **Key operations**: `.commit()` (save changes), `.reload()` (fetch latest)
- **Important**: Must explicit commit/reload, no automatic syncing
- **Use case**: Persist model weights, share data between functions

### **modal_secret_modal_docs.md**
- **Secret**: Injects environment variables securely
- **Creation**: Dashboard or `.from_name()`, `.from_dict()`, `.from_dotenv()`
- **Access**: Available as environment variables in container

## Web & API

### **modal_fastapi_endpoint_modal_docs.md**
- **@modal.fastapi_endpoint()**: Simple web endpoint wrapper
- **Auto CORS**: Enabled by default
- **Alternative**: Use `@modal.asgi_app` for full FastAPI app with multiple routes

## Real-World Example

### **how_to_deploy_stable_diffusion_3_5_large_on_modal_modal_blog.md**
- **Pattern**: Class with `@modal.enter()` to load model, `@modal.method()` for inference
- **GPU**: H100 for large models
- **Volume**: Cache HuggingFace models to speed up cold starts
- **Two @enter() methods**: One for loading weights, one for moving to GPU

---

## What We're Using in CapRemover

✅ **Used correctly:**
- `modal.App()` - App container
- `@app.cls()` - Class-based deployment
- `@modal.enter()` - Loading models on container start
- `@modal.method()` - Inference method
- `modal.Volume` - Caching model weights
- `modal.Secret` - R2 & AWS credentials
- `gpu="A100-80GB"` - GPU specification
- `timeout=900` - 15min timeout

⚠️ **Could improve:**
- Consider `scaledown_window` to keep containers warm longer (currently defaults to 60s)
- Could use `enable_memory_snapshot=True` for faster cold starts (after models loaded)
- Currently no `min_containers` - could add if need guaranteed availability

❌ **Not needed for now:**
- `buffer_containers` - For bursty traffic
- `@modal.exit()` - No cleanup needed
- `@modal.fastapi_endpoint()` - Using web endpoint already
