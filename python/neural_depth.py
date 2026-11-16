"""Pretrained monocular depth with Depth Anything V2 (via Hugging Face transformers).

Depth Anything V2 relative models output affine-invariant *inverse* depth
(disparity-like): larger values are nearer. This module returns the raw
prediction resized to the input grid; direction and scale are handled in
depth_alignment.py.
"""
from __future__ import annotations

import time
from pathlib import Path

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


class DepthPredictor:
    """Load DAV2 once, checking its output representation before use."""

    def __init__(self, model: str = "depth-anything-v2-small", device: str | None = None,
                 expected_type: str = "relative", cache_dir: str | None = None):
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForDepthEstimation
        from huggingface_hub import try_to_load_from_cache

        self.repo = MODELS.get(model, model)
        self.device = device or pick_device()
        config = AutoConfig.from_pretrained(self.repo, cache_dir=cache_dir)
        self.revision = getattr(config, "_commit_hash", None)
        cached_config = try_to_load_from_cache(self.repo, "config.json", cache_dir=cache_dir)
        if not self.revision and isinstance(cached_config, str):
            location = Path(cached_config)
            if location.parent.parent.name == "snapshots":
                self.revision = location.parent.name
        self.depth_type = config.depth_estimation_type
        if self.depth_type != expected_type:
            raise ValueError(f"Expected {expected_type} depth, but {self.repo} produces {self.depth_type}. Use the separate metric evaluation path for metric models.")
        self.processor = AutoImageProcessor.from_pretrained(self.repo, cache_dir=cache_dir)
        self.net = AutoModelForDepthEstimation.from_pretrained(self.repo, config=config, cache_dir=cache_dir).to(self.device).eval()

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        import torch
        from PIL import Image

        h, w = image_bgr.shape[:2]
        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        if self.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            pred = self.net(**inputs).predicted_depth
        if self.device == "mps":
            torch.mps.synchronize()
        runtime = time.perf_counter() - start
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
        depth = pred.squeeze().float().cpu().numpy()
        info = {
            "model": self.repo,
            "device": self.device,
            "inference_seconds": round(runtime, 4),
            "processor_input_shape": list(inputs["pixel_values"].shape),
            "output_convention": "metric depth in meters (larger = farther)" if self.depth_type == "metric" else "relative inverse depth (larger = nearer)",
            "depth_estimation_type": self.depth_type,
            "model_revision": self.revision,
            "model_max_depth": getattr(self.net.config, "max_depth", None),
        }
        return depth.astype(np.float32), info


def predict(image_bgr: np.ndarray, model: str = "depth-anything-v2-small", device: str | None = None) -> tuple[np.ndarray, dict]:
    """Return (raw inverse-depth prediction at the image resolution, run info)."""
    return DepthPredictor(model, device).predict(image_bgr)
