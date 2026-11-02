"""Synthetic focal stacks with known depth, for tests and pipeline smoke runs.

    python python/synthetic.py --output data/raw/synthetic_01

writes frame_*.png, manifest.json, and neural_depth.npy (a noisy, smoothed
inverse-depth map standing in for Depth Anything V2) so the full pipeline can
run without a phone or PyTorch:

    python python/run_pipeline.py --input data/raw/synthetic_01 --output results/synthetic_01 \
        --neural-depth data/raw/synthetic_01/neural_depth.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def textured_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.random((h, w)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 1.0)


def layered_scene(h: int = 240, w: int = 320, n_frames: int = 9, seed: int = 0):
    """Background wall (textured left half, flat right half) with two textured boxes in front.

    Returns (stack (N,H,W) float32 in [0,1], true focus index map (H,W), lens_positions).
    """
    lens_positions = np.linspace(0.1, 0.9, n_frames).astype(np.float32)
    texture = textured_image(h, w, seed)
    flat = np.full((h, w), 0.5, np.float32)
    true_idx = np.full((h, w), n_frames - 1.0, np.float32)  # wall at the farthest setting
    albedo = texture.copy()
    albedo[:, w // 2 :] = flat[:, w // 2 :]                 # textureless right half of the wall
    true_idx[h // 6 : h // 2, w // 8 : w // 3] = 1.0        # near box
    true_idx[h // 2 : 5 * h // 6, w // 2 : 3 * w // 4] = n_frames // 2  # middle box
    albedo[true_idx < n_frames - 1] = texture[true_idx < n_frames - 1]

    stack = []
    for k in range(n_frames):
        frame = np.zeros((h, w), np.float32)
        for layer in np.unique(true_idx):
            sigma = 0.01 + 1.2 * abs(layer - k)
            blurred = cv2.GaussianBlur(albedo, (0, 0), sigma) if sigma > 0.05 else albedo
            frame[true_idx == layer] = blurred[true_idx == layer]
        stack.append(frame)
    return np.stack(stack), true_idx, lens_positions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stack, true_idx, lens = layered_scene(n_frames=args.frames, seed=args.seed)

    frames = []
    for k, img in enumerate(stack):
        name = f"frame_{k:03d}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        frames.append({"file": name, "lens_position": float(lens[k]), "timestamp_ms": 400 * k})
    manifest = {"scene_id": out.name, "device": "synthetic", "camera": "synthetic", "frames": frames, "notes": "synthetic layered scene"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Stand-in for a monocular network: inverse depth, smoothed (soft edges), noisy, arbitrary affine.
    true_focus = np.interp(true_idx, np.arange(len(lens)), lens)
    inv = 1.0 / (0.2 + true_focus)
    inv = cv2.GaussianBlur(inv.astype(np.float32), (0, 0), 4.0)
    inv += np.random.default_rng(args.seed + 1).normal(0, 0.03, inv.shape).astype(np.float32)
    np.save(out / "neural_depth.npy", (3.0 * inv + 7.0).astype(np.float32))
    np.save(out / "true_focus.npy", true_focus.astype(np.float32))
    print(f"wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
