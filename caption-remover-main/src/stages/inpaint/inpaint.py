import os
import time
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DiffuEraser"))

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


def run_inpainting(video_path, mask_path, output_dir="data/results", video_length=None, mask_dilation=8, max_img_size=480, raft_iter=12, enable_pre_inference=False):
    """Run video inpainting using Propainter + DiffuEraser."""
    os.makedirs(output_dir, exist_ok=True)
    priori_path = os.path.join(output_dir, "propainter_result.mp4")
    final_path = os.path.join(output_dir, "diffueraser_result.mp4")
    
    base_dir = os.path.join(os.path.dirname(__file__), "DiffuEraser", "weights")
    base_model = os.path.join(base_dir, "stable-diffusion-v1-5")
    vae = os.path.join(base_dir, "sd-vae-ft-mse")
    diffueraser = os.path.join(base_dir, "diffuEraser")
    propainter = os.path.join(base_dir, "propainter")
    pcm_weights = os.path.join(base_dir, "PCM_Weights")
    
    print(f"\nStarting inpainting pipeline...")
    print(f"Input video: {video_path}")
    print(f"Input mask: {mask_path}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Check if model weights exist
    if not os.path.exists(os.path.join(propainter, "raft-things.pth")):
        print(f"WARNING: Propainter weights not found at {propainter}")
        print("Please download weights as described in DiffuEraser/weights/README.md")
    
    start = time.time()
    
    print("\n[1/2] Running Propainter...")
    print("Loading Propainter model (first run may take 1-2 minutes)...")
    propainter_model = Propainter(propainter, device=device)
    print("Propainter model loaded!")
    prop_start = time.time()
    propainter_model.forward(video_path, mask_path, priori_path, video_length=video_length, 
                            ref_stride=10, neighbor_length=10, subvideo_length=50, mask_dilation=mask_dilation, raft_iter=raft_iter)
    prop_time = time.time() - prop_start
    print(f"Propainter completed in {prop_time:.2f}s")
    
    print("\n[2/2] Running DiffuEraser...")
    diffueraser_model = DiffuEraser(device, base_model, vae, diffueraser, ckpt="2-Step", pcm_weights_path=pcm_weights)
    diff_start = time.time()
    diffueraser_model.forward(video_path, mask_path, priori_path, final_path,
                             max_img_size=max_img_size, video_length=video_length,
                             mask_dilation_iter=mask_dilation, guidance_scale=None,
                             enable_pre_inference=enable_pre_inference)
    diff_time = time.time() - diff_start
    print(f"DiffuEraser completed in {diff_time:.2f}s")
    
    total_time = time.time() - start
    print(f"\n=== Inpainting Summary ===")
    print(f"Propainter time: {prop_time:.2f}s")
    print(f"DiffuEraser time: {diff_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"\nOutputs:")
    print(f"  Propainter: {priori_path}")
    print(f"  DiffuEraser: {final_path}")
    
    torch.cuda.empty_cache()
    return priori_path, final_path

