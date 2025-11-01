import modal
import os
import sys
import time
from pathlib import Path

app = modal.App("caption-remover")
volume = modal.Volume.from_name("caption-remover-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("kashifmurtaza/caption-remover:modal-v3")
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
    def remove_captions(self, video_r2_url: str, model_to_use: str = "diffueraser") -> str:
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
            roi = (191, 1083, 508, 150)
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

            return result_url


@app.function(
    image=modal.Image.debian_slim().pip_install("fastapi[standard]"),
    timeout=3600  # 60 minutes - must match CaptionRemover timeout
)
@modal.fastapi_endpoint(method="POST")
def web(data: dict):
    video_url = data.get("video_url")
    model = data.get("model", "diffueraser")

    if not video_url:
        return {"error": "video_url is required"}, 400
    if model not in ["propainter", "diffueraser"]:
        return {"error": "model must be 'propainter' or 'diffueraser'"}, 400

    remover = CaptionRemover()
    result_url = remover.remove_captions.remote(video_url, model)

    return {"status": "success", "result_url": result_url, "model_used": model}


@app.local_entrypoint()
def main(video_url: str, model: str = "diffueraser"):
    remover = CaptionRemover()
    result = remover.remove_captions.remote(video_url, model)
    print(f"Result: {result}")
