import modal
import os
import sys
import time
from pathlib import Path

app = modal.App("caption-remover")
volume = modal.Volume.from_name("caption-remover-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("kashifmurtaza/caption-remover:modal-v3")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video"})  # Enable NVENC
    .add_local_dir("src", remote_path="/root/src", copy=False)
    .add_local_file("main.py", remote_path="/root/main.py", copy=False)
)


@app.cls(
    gpu="A100-80GB",
    timeout=3600,  # 60 minutes - increased for long videos
    image=image,
    volumes={"/weights": volume}, 
    secrets=[
        modal.Secret.from_name("r2-credentials"),
        modal.Secret.from_name("aws-s3-credentials")
    ]
)
class CaptionRemover:
    @modal.enter()
    def load_models(self):
        import torch

        print("CONTAINER STARTUP - LOADING MODELS")
        torch.cuda.init()

        sys.path.insert(0, "/root")
        sys.path.insert(0, "/root/src/stages/inpaint/DiffuEraser")

        weights_path = "/weights/DiffuEraser"
        if not os.path.exists(f"{weights_path}/propainter/raft-things.pth"):
            print("Downloading model weights from S3...")
            start = time.time()
            os.makedirs(weights_path, exist_ok=True)
            os.system(
                f"aws s3 sync s3://caption-remover2/caption-remover/src/stages/inpaint/DiffuEraser/ "
                f"{weights_path} --no-progress"
            )
            volume.commit()
            print(f"Weights downloaded in {time.time() - start:.1f}s")
        else:
            print("Using cached model weights from volume")

        from src.stages.inpaint.DiffuEraser.diffueraser.diffueraser import DiffuEraser
        from src.stages.inpaint.DiffuEraser.propainter.inference import Propainter, get_device

        print("Loading ProPainter model...")
        start = time.time()
        device = get_device()
        propainter_path = f"{weights_path}/propainter"
        self.propainter = Propainter(propainter_path, device=device)
        print(f"ProPainter loaded in {time.time() - start:.2f}s")

        print("Loading DiffuEraser model...")
        start = time.time()
        weights_subdir = f"{weights_path}/weights"
        base_model = f"{weights_subdir}/stable-diffusion-v1-5"
        vae = f"{weights_subdir}/sd-vae-ft-mse"
        diffueraser_path = f"{weights_subdir}/diffuEraser"
        pcm_weights = f"{weights_subdir}/PCM_Weights"

        self.diffueraser = DiffuEraser(
            device, base_model, vae, diffueraser_path,
            ckpt="2-Step", pcm_weights_path=pcm_weights
        )
        print(f"DiffuEraser loaded in {time.time() - start:.2f}s")

        self.device = device
        self.weights_path = weights_path
        print("MODELS LOADED - READY FOR INFERENCE")

    @modal.method()
    def remove_captions(
        self, 
        video_r2_url: str, 
        model_to_use: str = "diffueraser",
        roi_normalized: dict = None,
        webhook_url: str = None
    ) -> str:
        import boto3
        import tempfile
        import torch
        import torchvision
        from PIL import Image
        from src.stages.preprocessing import preprocess_video, run_precheck, crop_video_to_roi
        from src.stages.create_mask import create_mask_video
        from src.stages.postprocessing.composite import composite_inpainted_region

        print(f"PROCESSING VIDEO: {video_r2_url}")
        print(f"Model: {model_to_use}")

        total_start = time.time()
        r2_client = boto3.client(
            's3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY']
        )
        bucket_name = os.environ['R2_BUCKET_NAME']

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            data_dir = work_dir / "data"
            results_dir = data_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            print("Downloading video from R2...")
            download_start = time.time()
            video_key = video_r2_url.split('.r2.cloudflarestorage.com/')[-1]
            print(f"  Downloading: {bucket_name}/{video_key}")
            input_video = data_dir / "input.mp4"
            r2_client.download_file(bucket_name, video_key, str(input_video))
            print(f"Downloaded in {time.time() - download_start:.2f}s")

            max_resolution = 1600
            target_fps = 24.0
            raft_iter = 6
            enable_pre_inference = False

            print("Preprocessing video...")
            preprocess_start = time.time()
            processed_video = preprocess_video(
                video_path=str(input_video),
                max_resolution=max_resolution,
                target_fps=target_fps
            )
            info = run_precheck(
                video_path=processed_video,
                max_resolution=max_resolution,
                target_fps=target_fps
            )
            print(f"Preprocessed in {time.time() - preprocess_start:.2f}s")
            print(f"  Resolution: {info.width}x{info.height}, FPS: {info.fps}")

            # Convert normalized ROI to pixel coordinates
            if roi_normalized is None:
                # Default ROI (hardcoded value as normalized for 900x1600)
                roi_normalized = {"x": 0.212, "y": 0.677, "width": 0.564, "height": 0.094}
                print(f"Using default normalized ROI: {roi_normalized}")
            else:
                print(f"Using provided normalized ROI: {roi_normalized}")
            
            # Convert to pixel coordinates based on preprocessed video dimensions
            roi_x = int(roi_normalized["x"] * info.width)
            roi_y = int(roi_normalized["y"] * info.height)
            roi_w = int(roi_normalized["width"] * info.width)
            roi_h = int(roi_normalized["height"] * info.height)
            
            # Ensure even numbers (required for video encoding)
            roi_x = roi_x - (roi_x % 2)
            roi_y = roi_y - (roi_y % 2)
            roi_w = roi_w - (roi_w % 2)
            roi_h = roi_h - (roi_h % 2)
            
            roi = (roi_x, roi_y, roi_w, roi_h)
            print(f"  Pixel ROI: {roi} for {info.width}x{info.height} video")

            print("Creating mask...")
            mask_start = time.time()
            mask_video = create_mask_video(
                video_path=processed_video,
                debug=False,
                roi=roi
            )
            print(f"Mask created in {time.time() - mask_start:.2f}s")

            print("Cropping to 2x ROI region...")
            crop_start = time.time()
            cropped_video, new_roi = crop_video_to_roi(
                processed_video,
                roi,
                expansion_factor=2.0,
                output_path=str(results_dir / "video_cropped.mp4")
            )
            cropped_mask, _ = crop_video_to_roi(
                mask_video,
                roi,
                expansion_factor=2.0,
                output_path=str(results_dir / "mask_cropped.mp4")
            )
            print(f"Cropped in {time.time() - crop_start:.2f}s")

            print("Running inpainting...")
            inpaint_start = time.time()
            
            # Load frames in memory for optimization
            vframes, _, info = torchvision.io.read_video(filename=cropped_video, pts_unit='sec')
            video_frames = [Image.fromarray(f.numpy()) for f in vframes]
            video_fps = info['video_fps']
            mframes, _, _ = torchvision.io.read_video(filename=cropped_mask, pts_unit='sec')
            mask_frames = [Image.fromarray(f.numpy()) for f in mframes]

            # Step 1: Run ProPainter and write to disk (required for video codec processing)
            priori_path = str(results_dir / "propainter_result.mp4")
            self.propainter.forward(
                cropped_video, cropped_mask, priori_path,
                video_length=None,
                ref_stride=10, neighbor_length=10, subvideo_length=50,
                mask_dilation=8, raft_iter=raft_iter,
                preloaded_video_frames=video_frames,
                preloaded_video_fps=video_fps,
                preloaded_mask_frames=mask_frames
            )
            
            if model_to_use == "propainter":
                # ProPainter only mode
                inpaint_output = priori_path
                propainter_output = priori_path
                diffueraser_output = None
            else:
                # Step 2: Load ProPainter output from disk for DiffuEraser
                # (Video codec processing is essential for quality)
                priori_vframes, _, _ = torchvision.io.read_video(filename=priori_path, pts_unit='sec')
                priori_frames = [Image.fromarray(f.numpy()) for f in priori_vframes]
                
                # Step 3: Run DiffuEraser with disk-loaded priori frames
                diffueraser_path = str(results_dir / "diffueraser_result.mp4")
                self.diffueraser.forward(
                    cropped_video, cropped_mask, priori_path, diffueraser_path,
                    max_img_size=720, video_length=None,
                    mask_dilation_iter=8, guidance_scale=None,
                    enable_pre_inference=enable_pre_inference, nframes=22,
                    preloaded_video_frames=video_frames,
                    preloaded_video_fps=video_fps,
                    preloaded_mask_frames=mask_frames,
                    preloaded_priori_frames=priori_frames
                )
                inpaint_output = diffueraser_path
                propainter_output = priori_path
                diffueraser_output = diffueraser_path

            print(f"Inpainting completed in {time.time() - inpaint_start:.2f}s")
            print(f"Note: ProPainter written to disk for video codec processing (required for quality)")

            print("Compositing result...")
            composite_start = time.time()
            propainter_comp, diffueraser_comp = composite_inpainted_region(
                preprocessed_video=processed_video,
                roi=roi,
                propainter_result=propainter_output,
                diffueraser_result=diffueraser_output
            )
            final_output = diffueraser_comp if model_to_use == "diffueraser" else propainter_comp
            print(f"Composited in {time.time() - composite_start:.2f}s")

            print("Uploading result to R2...")
            upload_start = time.time()
            result_key = f"results/{Path(video_key).stem}_{model_to_use}_result.mp4"
            r2_client.upload_file(final_output, bucket_name, result_key)
            result_url = f"https://{bucket_name}.r2.cloudflarestorage.com/{result_key}"
            print(f"Uploaded in {time.time() - upload_start:.2f}s")

            torch.cuda.empty_cache()

            total_time = time.time() - total_start
            print(f"PROCESSING COMPLETE - {total_time:.2f}s ({total_time/60:.1f} min)")
            print(f"Result URL: {result_url}")

            # Call webhook if provided
            if webhook_url:
                try:
                    import requests
                    webhook_payload = {
                        "status": "success",
                        "result_url": result_url,
                        "model_used": model_to_use,
                        "processing_time": total_time,
                        "video_url": video_r2_url
                    }
                    webhook_response = requests.post(webhook_url, json=webhook_payload, timeout=30)
                    print(f"Webhook called: {webhook_url} (status: {webhook_response.status_code})")
                except Exception as e:
                    print(f"Webhook failed (non-critical): {e}")

            return result_url


# FastAPI app for handling all HTTP requests in a single container
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

web_app = FastAPI()


@web_app.post("/submit")
async def submit_job(request: Request):
    """Submit a video processing job."""
    data = await request.json()
    
    video_url = data.get("video_url")
    model = data.get("model", "diffueraser")
    roi = data.get("roi")
    webhook_url = data.get("webhook_url")

    # Validation
    if not video_url:
        return JSONResponse({"error": "video_url is required"}, status_code=400)
    if model not in ["propainter", "diffueraser"]:
        return JSONResponse({"error": "model must be 'propainter' or 'diffueraser'"}, status_code=400)
    
    # Validate ROI if provided
    if roi is not None:
        required_keys = ["x", "y", "width", "height"]
        if not all(k in roi for k in required_keys):
            return JSONResponse({"error": "roi must contain x, y, width, height"}, status_code=400)
        if not all(isinstance(roi[k], (int, float)) for k in required_keys):
            return JSONResponse({"error": "roi values must be numbers"}, status_code=400)
        if not all(0 <= roi[k] <= 1 for k in required_keys):
            return JSONResponse({"error": "roi values must be between 0 and 1"}, status_code=400)

    # Spawn async job (returns immediately)
    remover = CaptionRemover()
    call = remover.remove_captions.spawn(video_url, model, roi, webhook_url)

    return {
        "status": "processing",
        "job_id": call.object_id,
        "message": "Job submitted successfully. Use /status/{job_id} endpoint to check progress." + 
                   (" Webhook will be called when complete." if webhook_url else "")
    }


@web_app.get("/status/{job_id}")
async def check_status(job_id: str):
    """Check the status of a processing job."""
    from modal import FunctionCall
    
    try:
        call = FunctionCall.from_id(job_id)
        
        # Try to get result (non-blocking)
        try:
            result_url = call.get(timeout=0)
            return {
                "status": "completed",
                "result_url": result_url,
                "job_id": job_id
            }
        except TimeoutError:
            # Job still running - return 202 Accepted
            return JSONResponse(
                {
                    "status": "processing",
                    "job_id": job_id,
                    "message": "Job is still processing. Check again in a few seconds."
                },
                status_code=202
            )
            
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "job_id": job_id,
                "error": str(e)
            },
            status_code=500
        )


@app.function(
    image=modal.Image.debian_slim().pip_install("fastapi[standard]")
)
@modal.asgi_app()
def fastapi_app():
    """Single ASGI app handles all HTTP requests efficiently."""
    return web_app


@app.local_entrypoint()
def main(video_url: str, model: str = "diffueraser"):
    remover = CaptionRemover()
    result = remover.remove_captions.remote(video_url, model)
    print(f"Result: {result}")


