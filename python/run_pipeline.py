"""FocusDepth pipeline: focal stack -> optical depth + confidence -> neural depth -> fusion -> metrics.

    python python/run_pipeline.py --input data/raw/desk_01 --output results/desk_01 \
        --model depth-anything-v2-small --fusion all
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import all_in_focus
import confidence as conf_mod
import depth_alignment
import evaluate
import focus_depth
import fusion
import registration
import visualize

log = logging.getLogger("focusdepth")
REPO = Path(__file__).resolve().parent.parent


def load_manifest(scene_dir: Path, lens_field: str) -> tuple[dict, list[dict]]:
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    frames = manifest["frames"]
    for f in frames:
        # iOS manifests carry requested and measured positions; synthetic ones only "lens_position".
        f["_lens"] = float(f.get(lens_field, f.get("lens_position", f.get("lens_position_requested"))))
    frames = sorted(frames, key=lambda f: f["_lens"])
    lens = [f["_lens"] for f in frames]
    if len(set(lens)) != len(lens):
        raise ValueError(f"duplicate lens positions in manifest: {lens}")
    return manifest, frames


def load_images(scene_dir: Path, frames: list[dict], max_side: int) -> list[np.ndarray]:
    images = []
    for f in frames:
        img = cv2.imread(str(scene_dir / f["file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(scene_dir / f["file"])
        scale = max_side / max(img.shape[:2]) if max_side else 1.0
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(img)
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError(f"frames have different shapes: {shapes}")
    return images


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="scene folder with manifest.json and frames")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--neural-input", choices=["reference", "aif"], default="reference")
    ap.add_argument("--model", default=None, help="override neural.model from the config")
    ap.add_argument("--fusion", choices=["all", "fixed", "confidence"], default="all")
    ap.add_argument("--neural-depth", type=Path, help="precomputed raw neural .npy (skips PyTorch)")
    ap.add_argument("--annotations", type=Path, help="defaults to data/annotations/<scene_id>.json if present")
    ap.add_argument("--alignment-params", type=Path, help="reuse a scale/shift JSON instead of fitting per scene")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    tick = time.perf_counter

    # --- load and register ---------------------------------------------------
    t = tick()
    manifest, frames = load_manifest(args.input, cfg["capture"]["lens_position_field"])
    images = load_images(args.input, frames, cfg["capture"]["max_side"])
    lens = np.array([f["_lens"] for f in frames], np.float32)
    ref = len(images) // 2 if cfg["registration"]["reference"] == "middle" else int(cfg["registration"]["reference"])
    timings["load"] = tick() - t

    t = tick()
    reg_cfg = cfg["registration"]
    if reg_cfg["enabled"]:
        reg = registration.register_stack(
            images, ref, reg_cfg["motion"], reg_cfg["iterations"], float(reg_cfg["eps"]), pyramid_levels=reg_cfg["pyramid_levels"]
        )
    else:
        h, w = images[0].shape[:2]
        reg = registration.RegistrationResult(np.stack(images), [np.eye(2, 3)] * len(images), [1.0] * len(images), (0, 0, w, h))
    timings["registration"] = tick() - t
    x0, y0, x1, y1 = reg.crop
    log.info("registered %d frames, crop=%s, failed=%s", len(images), reg.crop, reg.failed)

    reference = reg.images[ref]
    cv2.imwrite(str(out / "reference_full.png"), images[ref])
    cv2.imwrite(str(out / "reference_rgb.png"), reference)

    # --- optical depth and confidence -----------------------------------------
    t = tick()
    o_cfg = cfg["optical"]
    gray = np.stack([registration.to_gray_float(im) for im in reg.images])
    sharp = focus_depth.sharpness_stack(gray, o_cfg["laplacian_ksize"], o_cfg["smooth_sigma"])
    d_opt = focus_depth.optical_depth(sharp, lens, o_cfg["method"])
    c_cfg = cfg["confidence"]
    conf = conf_mod.optical_confidence(sharp, c_cfg["measure"], c_cfg["texture_gate"], c_cfg["gate_low"], c_cfg["gate_high"])
    timings["optical"] = tick() - t
    log.info("optical depth and confidence done")

    if reg.failed:
        if args.neural_input == "aif":
            raise ValueError(f"Cannot use AIF neural input: registration failed for frames {reg.failed}")
        log.warning("Exported AIF is diagnostic only: registration failed for frames %s", reg.failed)
    aif, labels = all_in_focus.compose(reg.images, sharp, ref)
    cv2.imwrite(str(out / "all_in_focus.png"), aif)
    np.save(out / "aif_source_indices.npy", labels)
    neural_image = aif if args.neural_input == "aif" else reference

    # --- neural depth ----------------------------------------------------------
    t = tick()
    neural_info: dict = {}
    if args.neural_depth:
        raw = np.load(args.neural_depth).astype(np.float32)
        if raw.shape != images[ref].shape[:2]:
            raw = cv2.resize(raw, (images[ref].shape[1], images[ref].shape[0]), interpolation=cv2.INTER_LINEAR)
        raw = raw[y0:y1, x0:x1]
        neural_info = {"source": str(args.neural_depth)}
    else:
        import neural_depth

        raw, neural_info = neural_depth.predict(neural_image, args.model or cfg["neural"]["model"])
    timings["neural"] = tick() - t
    log.info("neural depth done")

    # --- alignment -----------------------------------------------------------
    a_cfg = cfg["alignment"]
    params = None
    if args.alignment_params:
        params = depth_alignment.AlignmentParams(**json.loads(args.alignment_params.read_text()))
    d_nn, params = depth_alignment.align_neural_to_optical(raw, d_opt, conf, a_cfg["conf_threshold"], a_cfg["min_pixels"], params=params)
    if params.scale > 0:
        log.warning("alignment scale is positive: raw neural output increases with distance, unlike Depth Anything V2's usual inverse-depth output. Check conventions.")

    # --- fusion --------------------------------------------------------------
    maps = {"B1 optical": d_opt, "B2 neural": d_nn}
    if args.fusion in ("all", "fixed"):
        maps["B3 fixed fusion"] = fusion.fixed_fusion(d_opt, d_nn, cfg["fusion"]["fixed_weight"])
    if args.fusion in ("all", "confidence"):
        maps["Confidence fusion"] = fusion.confidence_fusion(d_opt, d_nn, conf)

    # --- outputs -------------------------------------------------------------
    files = {"B1 optical": "optical_depth", "B2 neural": "neural_depth", "B3 fixed fusion": "fixed_fusion", "Confidence fusion": "confidence_fusion"}
    # Same shared range as the comparison figure, so every depth PNG uses comparable colours.
    vmin, vmax = (float(v) for v in np.percentile(np.concatenate([d.ravel() for d in maps.values()]), [1, 99]))
    for name, d in maps.items():
        np.save(out / f"{files[name]}.npy", d)
        visualize.save_depth_png(out / f"{files[name]}.png", d, vmin, vmax)
    np.save(out / "neural_depth_raw.npy", raw)
    np.save(out / "optical_confidence.npy", conf)
    visualize.save_confidence_png(out / "optical_confidence.png", conf)
    visualize.comparison_figure(out / "comparison.png", reference, maps, conf, manifest.get("scene_id", ""))

    metrics = {
        "scene_id": manifest.get("scene_id"),
        "n_frames": len(frames),
        "lens_positions": lens.tolist(),
        "reference_index": ref,
        "image_shape": list(reference.shape[:2]),
        "registration": {"crop_xyxy": list(reg.crop), "ecc_scores": reg.ecc_scores, "failed": reg.failed},
        "neural": {**neural_info, "input": args.neural_input},
        "alignment": params.to_dict(),
        "direction_check": {
            "spearman_raw_neural_vs_optical": depth_direction_raw(d_opt, raw, conf, a_cfg["conf_threshold"]),
            "expected": "negative for inverse-depth model output",
        },
        "confidence": {"mean": float(conf.mean()), "frac_above_threshold": float((conf >= a_cfg["conf_threshold"]).mean())},
        "timings_s": {k: round(v, 4) for k, v in timings.items()},
        "config": cfg,
    }

    ann_path = args.annotations or REPO / "data" / "annotations" / f"{manifest.get('scene_id')}.json"
    if ann_path.exists():
        ann = evaluate.load_annotations(ann_path)
        metrics["evaluation"] = evaluate.evaluate_scene(maps, conf, ann, (x0, y0), cfg["evaluation"]["min_separation_m"])
        for name, res in metrics["evaluation"]["ordering"].items():
            log.info("ordering accuracy %-18s %.3f (%d pairs)", name, res["accuracy"], res["n_pairs"])
    else:
        log.info("no annotations at %s; skipping ordering evaluation", ann_path)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("wrote results to %s", out)


def depth_direction_raw(d_opt, raw, conf, threshold) -> float:
    return depth_alignment.depth_direction(d_opt, raw, conf >= threshold)


if __name__ == "__main__":
    main()
