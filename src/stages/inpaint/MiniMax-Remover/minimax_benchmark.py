import os
import time
import argparse
import numpy as np
import torch

from decord import VideoReader
from diffusers.utils import export_to_video
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

# Local modules from MiniMax-Remover
from transformer_minimax_remover import Transformer3DModel
from pipeline_minimax_remover import Minimax_Remover_Pipeline


def load_video_frames(path: str) -> np.ndarray:
    """Load video as float32 frames in [-1, 1], shape [T, H, W, 3]."""
    vr = VideoReader(path)
    frames = vr.get_batch(range(len(vr))).asnumpy()  # uint8 [T,H,W,3]
    frames = frames.astype(np.float32) / 127.5 - 1.0
    return frames


def load_mask_frames(path: str, num_frames: int) -> np.ndarray:
    """Load mask video and convert to [0,1] grayscale mask with shape [T, H, W, 1].

    If mask has fewer frames than input video, repeat last frame.
    """
    vr = VideoReader(path)
    masks = vr.get_batch(range(min(len(vr), num_frames))).asnumpy()  # [T,H,W,3]
    if masks.shape[0] < num_frames:
        pad = np.repeat(masks[-1:,...], num_frames - masks.shape[0], axis=0)
        masks = np.concatenate([masks, pad], axis=0)
    # Convert to grayscale and normalize to [0,1]
    masks = masks.mean(axis=-1, keepdims=True) / 255.0
    # Binarize
    masks = (masks > 0.5).astype(np.float32)
    return masks


def main():
    parser = argparse.ArgumentParser(description="MiniMax-Remover benchmark runner")
    parser.add_argument("--input_video", type=str, default="../../../data/examples/34s1080_inpaint.mp4",
                        help="Path to input video (downscaled recommended)")
    parser.add_argument("--input_mask", type=str, default="../../../data/results/mask_video_inpaint.mp4",
                        help="Path to mask video (same fps/length as input)")
    parser.add_argument("--out", type=str, default="../../../data/results/minimax_result.mp4",
                        help="Output video path")
    parser.add_argument("--steps", type=int, default=12, help="num_inference_steps (6-12 typical)")
    parser.add_argument("--iterations", type=int, default=6, help="iterations per README (fast default)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load inputs
    vid = load_video_frames(args.input_video)  # [-1,1], [T,H,W,3]
    T, H, W, _ = vid.shape
    msk = load_mask_frames(args.input_mask, T)  # [0,1], [T,H,W,1]

    # Model folders (downloaded into this directory per README)
    base_dir = os.path.dirname(__file__)
    vae_dir = os.path.join(base_dir, "vae")
    transformer_dir = os.path.join(base_dir, "transformer")
    scheduler_dir = os.path.join(base_dir, "scheduler")

    if not (os.path.isdir(vae_dir) and os.path.isdir(transformer_dir) and os.path.isdir(scheduler_dir)):
        raise FileNotFoundError(
            "MiniMax weights not found. Ensure 'vae/', 'transformer/', and 'scheduler/' exist in this folder."
        )

    # Load weights
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained(transformer_dir, torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained(scheduler_dir)

    pipe = Minimax_Remover_Pipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(device)

    gen = torch.Generator(device=device).manual_seed(42)

    print("MiniMax-Remover inference...")
    start = time.time()
    result = pipe(
        images=vid,              # np.ndarray [-1,1], [T,H,W,3]
        masks=msk,               # np.ndarray [0,1], [T,H,W,1]
        num_frames=T,
        height=H,
        width=W,
        num_inference_steps=args.steps,
        generator=gen,
        iterations=args.iterations,
    ).frames[0]
    elapsed = time.time() - start
    print(f"MiniMax-Remover completed in {elapsed:.2f}s ({elapsed/60:.1f} min)")

    # Save
    out_path = os.path.abspath(os.path.join(base_dir, args.out)) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_to_video(result, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


