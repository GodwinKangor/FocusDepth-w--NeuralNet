"""Register a focal stack to a reference frame with OpenCV ECC.

Focus breathing slightly changes magnification between focus positions, so the
default motion model is affine rather than pure translation. After warping,
every frame is cropped to the rectangle where all frames have valid pixels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

MOTION_MODELS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
}


@dataclass
class RegistrationResult:
    images: np.ndarray                 # (N, h, w, C) or (N, h, w), cropped
    warps: list[np.ndarray]            # 2x3 warps mapping reference -> frame coordinates
    ecc_scores: list[float]
    crop: tuple[int, int, int, int]    # (x0, y0, x1, y1) in reference-frame pixels
    failed: list[int] = field(default_factory=list)


def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img / 255.0 if img.max() > 1.0 else img


def _valid_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest centred rectangle whose rows and columns are fully valid."""
    h, w = mask.shape
    y0, y1, x0, x1 = 0, h, 0, w
    # Shrink the box from whichever side still contains invalid pixels.
    while y1 - y0 > 2 and x1 - x0 > 2:
        sub = mask[y0:y1, x0:x1]
        if sub.all():
            break
        bad = ~sub
        top, bottom = bad[0].sum(), bad[-1].sum()
        left, right = bad[:, 0].sum(), bad[:, -1].sum()
        worst = max(top, bottom, left, right)
        if worst == top:
            y0 += 1
        elif worst == bottom:
            y1 -= 1
        elif worst == left:
            x0 += 1
        else:
            x1 -= 1
    return x0, y0, x1, y1


def register_stack(
    images: list[np.ndarray] | np.ndarray,
    ref_index: int,
    motion: str = "affine",
    iterations: int = 200,
    eps: float = 1e-6,
    gauss_filt_size: int = 5,
    max_side_for_ecc: int = 1024,
    pyramid_levels: int = 3,
) -> RegistrationResult:
    """Align every frame to images[ref_index].

    Frames far from the reference in focus look very different (heavy defocus),
    so each frame is registered to its neighbour one step closer to the
    reference and the warps are chained. ECC runs coarse-to-fine on downscaled
    copies (fine texture alone traps it in local minima); translation terms are
    rescaled to full resolution at the end. A link where ECC fails falls back to
    the identity, and the frame is listed in ``failed`` for inspection.
    """
    images = [np.asarray(im) for im in images]
    h, w = images[0].shape[:2]
    scale = min(1.0, max_side_for_ecc / max(h, w))
    small = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else im

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    motion_type = MOTION_MODELS[motion]

    def pyramid(img: np.ndarray) -> list[np.ndarray]:
        levels = [small(to_gray_float(img))]
        for _ in range(pyramid_levels - 1):
            levels.append(cv2.pyrDown(levels[-1]))
        return levels[::-1]  # coarsest first

    pyramids = [pyramid(im) for im in images]
    n = len(images)
    # 3x3 homogeneous warps mapping reference coordinates -> frame coordinates (ECC scale).
    to_frame = [np.eye(3) for _ in range(n)]
    scores = [1.0] * n
    failed: list[int] = []
    order = list(range(ref_index - 1, -1, -1)) + list(range(ref_index + 1, n))
    for i in order:
        j = i + 1 if i < ref_index else i - 1  # neighbour that is already registered
        link = np.eye(2, 3, dtype=np.float32)
        try:
            for level, (tmpl, inp) in enumerate(zip(pyramids[j], pyramids[i])):
                if level > 0:
                    link[:, 2] *= 2.0
                scores[i], link = cv2.findTransformECC(tmpl, inp, link, motion_type, criteria, None, gauss_filt_size)
        except cv2.error as exc:
            log.warning("ECC failed between frames %d and %d (%s); using identity link", j, i, exc.err)
            failed.append(i)
            link = np.eye(2, 3, dtype=np.float32)
            scores[i] = float("nan")
        to_frame[i] = np.vstack([link, [0, 0, 1]]) @ to_frame[j]

    aligned, warps = [], []
    valid = np.ones((h, w), dtype=bool)
    for img, m in zip(images, to_frame):
        warp = m[:2].astype(np.float32)
        warp[:, 2] /= scale
        flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        aligned.append(cv2.warpAffine(img, warp, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT))
        frame_valid = cv2.warpAffine(np.ones((h, w), np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        valid &= frame_valid.astype(bool)
        warps.append(warp)

    x0, y0, x1, y1 = _valid_crop(valid)
    stack = np.stack([im[y0:y1, x0:x1] for im in aligned])
    return RegistrationResult(stack, warps, scores, (x0, y0, x1, y1), failed)
