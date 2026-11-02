"""Ordinal evaluation against hand-measured object distances.

Annotation file (data/annotations/<scene_id>.json):

    {
      "scene_id": "desk_01",
      "regions": [
        {"name": "mug", "bbox": [x0, y0, x1, y1], "distance_m": 0.42, "type": "textured"},
        {"name": "wall", "bbox": [...], "distance_m": 1.30, "type": "textureless"}
      ]
    }

Bounding boxes are in pixels of the *reference frame after resizing* (the
image saved as reference_full.png), before the registration crop. Distances are
measured independently (tape measure) and are used only for ordering.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def load_annotations(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def region_values(depth: np.ndarray, bbox, crop_offset=(0, 0)) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    ox, oy = crop_offset
    h, w = depth.shape
    xs = slice(max(0, x0 - ox), min(w, x1 - ox))
    ys = slice(max(0, y0 - oy), min(h, y1 - oy))
    vals = depth[ys, xs]
    if vals.size == 0:
        raise ValueError(f"region {bbox} falls outside the cropped image")
    return vals.ravel()


def pairwise_ordering(depth: np.ndarray, regions: list[dict], crop_offset=(0, 0), min_separation_m: float = 0.05) -> dict:
    """Fraction of region pairs whose median depths are ordered like the measured distances."""
    medians = {r["name"]: float(np.median(region_values(depth, r["bbox"], crop_offset))) for r in regions}
    correct, total, pairs = 0, 0, []
    for a, b in itertools.combinations(regions, 2):
        dz = b["distance_m"] - a["distance_m"]
        if abs(dz) < min_separation_m:
            continue
        ok = np.sign(medians[b["name"]] - medians[a["name"]]) == np.sign(dz)
        correct += int(ok)
        total += 1
        pairs.append({"near": a["name"] if dz > 0 else b["name"], "far": b["name"] if dz > 0 else a["name"], "correct": bool(ok)})
    return {"accuracy": correct / total if total else float("nan"), "n_pairs": total, "region_medians": medians, "pairs": pairs}


def region_stats(confidence: np.ndarray, regions: list[dict], crop_offset=(0, 0)) -> dict:
    return {
        r["name"]: {
            "type": r.get("type", ""),
            "mean_confidence": float(region_values(confidence, r["bbox"], crop_offset).mean()),
        }
        for r in regions
    }


def evaluate_scene(maps: dict[str, np.ndarray], confidence: np.ndarray, annotations: dict, crop_offset=(0, 0), min_separation_m: float = 0.05) -> dict:
    regions = annotations["regions"]
    return {
        "ordering": {name: pairwise_ordering(d, regions, crop_offset, min_separation_m) for name, d in maps.items()},
        "regions": region_stats(confidence, regions, crop_offset),
    }
