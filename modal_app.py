"""
Modal deployment for Caption Remover
Removes captions from videos using ProPainter + DiffuEraser models
"""

import modal
import os
import sys
import time
from pathlib import Path

# Create Modal app
app = modal.App("caption-remover")

# Create persistent volume for model weights
volume = modal.Volume.from_name("caption-remover-weights", create_if_missing=True)

# Build Docker image following install.sh
image = (
    # Use PaddlePaddle's official Docker image - includes PaddlePaddle 3.2.0 + CUDA 12.6 + cuDNN 9
    # Note: Don't add_python - use the Python version that comes with the image
    modal.Image.from_registry(
        "paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5"
    )
    # System dependencies
    .apt_install("ffmpeg")
    .run_commands("conda remove -y --force ffmpeg 2>/dev/null || true")
    
    # PaddlePaddle 3.2.0 already included in base image ✓
    
    # OpenCV and PaddleOCR
    .pip_install(
        "opencv-python",
        "paddleocr[all]",
        "langchain==0.0.354",  # For PaddleX compatibility
        "numpy<2",
        "decord",
        "diffusers==0.29.2",
        "transformers==4.41.1",
        "matplotlib",
        "fastapi[standard]"
    )
    
    # DiffuEraser dependencies
    .pip_install(
        "av==14.0.1",
        "accelerate==0.25.0",
        "imageio==2.34.1",
        "einops==0.8.0",
        "datasets==2.19.1",
        "peft==0.13.2",
        "scipy==1.13.1"
    )
    
    # PyTorch for CUDA 11.8 (matches working install.sh)
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    
    # Fix cuDNN conflict: PyTorch needs cuDNN 8, PaddlePaddle needs cuDNN 9
    # Base image has cuDNN 9, we add cuDNN 8 files for PyTorch
    .run_commands(
        "cd /tmp && "
        "pip download --no-deps nvidia-cudnn-cu11==8.7.0.84 && "
        "python3 -c \"import zipfile; zipfile.ZipFile('nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl').extractall()\" && "
        "cp nvidia/cudnn/lib/libcudnn*.so.8 /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/ 2>/dev/null || "
        "mkdir -p /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/ && "
        "cp nvidia/cudnn/lib/libcudnn*.so.8 /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/ && "
        "rm -rf nvidia *.whl"
    )
    
    # AWS CLI for downloading weights
    .pip_install("awscli", "boto3")
    
    # Add source code
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("main.py", remote_path="/root/main.py")
)


@app.cls(
    gpu="A100-80GB",
    timeout=900,  # 15 min timeout
    image=image,
    volumes={"/weights": volume},
    secrets=[
        modal.Secret.from_name("r2-credentials"),
        modal.Secret.from_name("aws-s3-credentials")
    ]
)
class CaptionRemover:
    """Caption removal service using ProPainter + DiffuEraser"""
    
    @modal.enter()
    def load_models(self):
        """Download weights and load models into GPU memory (runs once per container)"""
        import torch
        
        print("=" * 60)
        print("CONTAINER STARTUP - LOADING MODELS")
        print("=" * 60)
        
        # Force CUDA initialization before PaddlePaddle
        torch.cuda.init()
        
        # Add src to Python path
        sys.path.insert(0, "/root")
        sys.path.insert(0, "/root/src/stages/inpaint/DiffuEraser")
        
        # Download weights if not cached
        weights_path = "/weights/DiffuEraser"
        if not os.path.exists(f"{weights_path}/propainter/raft-things.pth"):
            print("\n[1/3] Downloading model weights from S3 (first time only)...")
            print("This will take 5-10 minutes but only happens once...")
            start = time.time()
            
            os.makedirs(weights_path, exist_ok=True)
            os.system(
                f"aws s3 sync s3://caption-remover2/caption-remover/src/stages/inpaint/DiffuEraser/ "
                f"{weights_path} --no-progress"
            )
            
            # Commit to volume so it persists
            volume.commit()
            print(f"✓ Weights downloaded in {time.time() - start:.1f}s")
        else:
            print("\n[1/3] Using cached model weights from volume ✓")
        
        # Import after weights are ready
        from src.stages.inpaint.DiffuEraser.diffueraser.diffueraser import DiffuEraser
        from src.stages.inpaint.DiffuEraser.propainter.inference import Propainter, get_device
        
        # Load models into GPU memory
        print("\n[2/3] Loading ProPainter model into GPU...")
        start = time.time()
        device = get_device()
        propainter_path = f"{weights_path}/propainter"
        self.propainter = Propainter(propainter_path, device=device)
        print(f"✓ ProPainter loaded in {time.time() - start:.2f}s")
        
        print("\n[3/3] Loading DiffuEraser model into GPU...")
        start = time.time()
        # Models are in weights subdirectory
        weights_subdir = f"{weights_path}/weights"
        base_model = f"{weights_subdir}/stable-diffusion-v1-5"
        vae = f"{weights_subdir}/sd-vae-ft-mse"
        diffueraser_path = f"{weights_subdir}/diffuEraser"
        pcm_weights = f"{weights_subdir}/PCM_Weights"
        
        self.diffueraser = DiffuEraser(
            device, base_model, vae, diffueraser_path,
            ckpt="2-Step", pcm_weights_path=pcm_weights
        )
        print(f"✓ DiffuEraser loaded in {time.time() - start:.2f}s")
        
        self.device = device
        self.weights_path = weights_path
        
        print("\n" + "=" * 60)
        print("✓ MODELS LOADED - READY FOR INFERENCE")
        print("=" * 60)
    
    @modal.method()
    def remove_captions(self, video_r2_url: str, model_to_use: str = "diffueraser") -> str:
        """
        Remove captions from video
        
        Args:
            video_r2_url: R2 URL of input video
            model_to_use: "propainter" or "diffueraser"
        
        Returns:
            R2 URL of processed video
        """
        import boto3
        import tempfile
        import torch
        import torchvision
        from PIL import Image
        
        # Import your pipeline stages
        from src.stages.preprocessing import preprocess_video, run_precheck, crop_video_to_roi
        from src.stages.create_mask import create_mask_video
        from src.stages.postprocessing.composite import composite_inpainted_region
        
        print("\n" + "=" * 60)
        print(f"PROCESSING VIDEO: {video_r2_url}")
        print(f"Model: {model_to_use}")
        print("=" * 60)
        
        total_start = time.time()
        
        # Initialize R2 client
        r2_client = boto3.client(
            's3',
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY']
        )
        bucket_name = os.environ['R2_BUCKET_NAME']
        
        # Create temp working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            data_dir = work_dir / "data"
            results_dir = data_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Download video from R2
            print("\n[1/6] Downloading video from R2...")
            download_start = time.time()
            # Extract S3 key from URL (format: https://bucket.r2.cloudflarestorage.com/path/to/file)
            video_key = video_r2_url.split('.r2.cloudflarestorage.com/')[-1]
            print(f"  Downloading: {bucket_name}/{video_key}")
            input_video = data_dir / "input.mp4"
            r2_client.download_file(bucket_name, video_key, str(input_video))
            print(f"✓ Downloaded in {time.time() - download_start:.2f}s")
            
            # Hardcoded configuration (from main.py)
            max_resolution = 1600
            target_fps = 24.0
            roi = (191, 1083, 508, 150)
            raft_iter = 6
            enable_pre_inference = False
            
            # [2/6] Preprocessing
            print("\n[2/6] Preprocessing video...")
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
            print(f"✓ Preprocessed in {time.time() - preprocess_start:.2f}s")
            print(f"  Resolution: {info.width}x{info.height}, FPS: {info.fps}")
            
            # [3/6] Create mask
            print("\n[3/6] Creating mask...")
            mask_start = time.time()
            mask_video = create_mask_video(
                video_path=processed_video,
                debug=False,
                roi=roi
            )
            print(f"✓ Mask created in {time.time() - mask_start:.2f}s")
            
            # [4/6] Crop to 2x ROI region
            print("\n[4/6] Cropping to 2x ROI region...")
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
            print(f"✓ Cropped in {time.time() - crop_start:.2f}s")
            
            # [5/6] Inpainting
            print("\n[5/6] Running inpainting...")
            inpaint_start = time.time()
            
            # Load frames in memory
            vframes, _, info = torchvision.io.read_video(filename=cropped_video, pts_unit='sec')
            video_frames = [Image.fromarray(f.numpy()) for f in vframes]
            video_fps = info['video_fps']
            
            mframes, _, _ = torchvision.io.read_video(filename=cropped_mask, pts_unit='sec')
            mask_frames = [Image.fromarray(f.numpy()) for f in mframes]
            
            # Run ProPainter
            if model_to_use == "propainter":
                # ProPainter only mode
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
                inpaint_output = priori_path
                propainter_output = priori_path
                diffueraser_output = None
            else:
                # DiffuEraser mode (ProPainter + DiffuEraser)
                priori_frames = self.propainter.forward_in_memory(
                    cropped_video, cropped_mask,
                    video_length=None,
                    ref_stride=10, neighbor_length=10, subvideo_length=50,
                    mask_dilation=8, raft_iter=raft_iter,
                    preloaded_video_frames=video_frames,
                    preloaded_video_fps=video_fps,
                    preloaded_mask_frames=mask_frames
                )
                
                # Run DiffuEraser
                diffueraser_path = str(results_dir / "diffueraser_result.mp4")
                self.diffueraser.forward(
                    cropped_video, cropped_mask, None, diffueraser_path,
                    max_img_size=720, video_length=None,
                    mask_dilation_iter=8, guidance_scale=None,
                    enable_pre_inference=enable_pre_inference, nframes=22,
                    preloaded_video_frames=video_frames,
                    preloaded_video_fps=video_fps,
                    preloaded_mask_frames=mask_frames,
                    preloaded_priori_frames=priori_frames
                )
                inpaint_output = diffueraser_path
                propainter_output = None
                diffueraser_output = diffueraser_path
            
            print(f"✓ Inpainting completed in {time.time() - inpaint_start:.2f}s")
            
            # [6/6] Composite back onto original
            print("\n[6/6] Compositing result...")
            composite_start = time.time()
            propainter_comp, diffueraser_comp = composite_inpainted_region(
                preprocessed_video=processed_video,
                roi=roi,
                propainter_result=propainter_output,
                diffueraser_result=diffueraser_output
            )
            
            final_output = diffueraser_comp if model_to_use == "diffueraser" else propainter_comp
            print(f"✓ Composited in {time.time() - composite_start:.2f}s")
            
            # Upload result to R2
            print("\n[7/7] Uploading result to R2...")
            upload_start = time.time()
            result_key = f"results/{Path(video_key).stem}_{model_to_use}_result.mp4"
            r2_client.upload_file(final_output, bucket_name, result_key)
            result_url = f"https://{bucket_name}.r2.cloudflarestorage.com/{result_key}"
            print(f"✓ Uploaded in {time.time() - upload_start:.2f}s")
            
            # Cleanup GPU memory
            torch.cuda.empty_cache()
            
            total_time = time.time() - total_start
            print("\n" + "=" * 60)
            print(f"✓ PROCESSING COMPLETE - {total_time:.2f}s ({total_time/60:.1f} min)")
            print(f"Result URL: {result_url}")
            print("=" * 60)
            
            return result_url


# HTTP endpoint - calls the GPU class on-demand
@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint(method="POST")
def web(data: dict):
    """
    HTTP API endpoint for on-demand inference
    
    POST body:
    {
        "video_url": "https://...",
        "model": "diffueraser"  // or "propainter"
    }
    """
    video_url = data.get("video_url")
    model = data.get("model", "diffueraser")
    
    if not video_url:
        return {"error": "video_url is required"}, 400
    
    if model not in ["propainter", "diffueraser"]:
        return {"error": "model must be 'propainter' or 'diffueraser'"}, 400
    
    # Call the GPU class - Modal will spin up GPU container on-demand
    remover = CaptionRemover()
    result_url = remover.remove_captions.remote(video_url, model)
    
    return {
        "status": "success",
        "result_url": result_url,
        "model_used": model
    }


# Local entrypoint for testing
@app.local_entrypoint()
def main(video_url: str, model: str = "diffueraser"):
    """Test locally: modal run modal_app.py --video-url <url> --model diffueraser"""
    remover = CaptionRemover()
    result = remover.remove_captions.remote(video_url, model)
    print(f"\n✨ Result: {result}")
