"""Colour-mapped depth images and the side-by-side comparison figure."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def depth_to_color(d: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Turbo colour map; near = warm, far = cool (assumes larger = farther)."""
    vmin = np.percentile(d, 1) if vmin is None else vmin
    vmax = np.percentile(d, 99) if vmax is None else vmax
    t = np.clip((d - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    return cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def save_depth_png(path: str | Path, d: np.ndarray, vmin=None, vmax=None) -> None:
    cv2.imwrite(str(path), depth_to_color(d, vmin, vmax))


def save_confidence_png(path: str | Path, c: np.ndarray) -> None:
    cv2.imwrite(str(path), (np.clip(c, 0, 1) * 255).astype(np.uint8))


def comparison_figure(path: str | Path, reference_bgr: np.ndarray, maps: dict[str, np.ndarray], confidence: np.ndarray, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Shared colour range so the depth panels are directly comparable.
    stacked = np.concatenate([m.ravel() for m in maps.values()])
    vmin, vmax = np.percentile(stacked, [1, 99])

    panels = [("Reference RGB", cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB), None)]
    panels += [(name, m, "turbo_r") for name, m in maps.items()]
    panels.append(("Optical confidence", confidence, "gray"))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, img, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(img)
        elif cmap == "gray":
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
