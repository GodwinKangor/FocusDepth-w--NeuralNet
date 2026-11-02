"""Pretrained monocular depth with Depth Anything V2 (via Hugging Face transformers).

Depth Anything V2 relative models output affine-invariant *inverse* depth
(disparity-like): larger values are nearer. This module returns the raw
prediction resized to the input grid; direction and scale are handled in
depth_alignment.py.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

MODELS = {
    "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything-v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything-v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def predict(image_bgr: np.ndarray, model: str = "depth-anything-v2-small", device: str | None = None) -> tuple[np.ndarray, dict]:
    """Return (raw inverse-depth prediction at the image's resolution, run info)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo = MODELS.get(model, model)
    device = device or pick_device()
    processor = AutoImageProcessor.from_pretrained(repo)
    net = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()

    h, w = image_bgr.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=rgb, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        pred = net(**inputs).predicted_depth
    if device == "mps":
        torch.mps.synchronize()
    runtime = time.perf_counter() - start

    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
    depth = pred.squeeze().float().cpu().numpy()
    info = {
        "model": repo,
        "device": device,
        "inference_seconds": round(runtime, 4),
        "processor_input_shape": list(inputs["pixel_values"].shape),
        "output_convention": "relative inverse depth (larger = nearer)",
    }
    return depth.astype(np.float32), info
