from __future__ import annotations

import numpy as np


def _cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for stereo matching helpers. Install the notebook "
            "requirements with: python -m pip install -r requirements.txt"
        ) from exc
    return cv2


def to_gray_float(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.float32) / 255.0


def sad_patch_cost(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    y: int,
    x: int,
    disparity: int,
    radius: int,
) -> float:
    """Student TODO: compute sum of absolute differences for one pixel/disparity."""
    raise NotImplementedError("Implement this in the notebook.")


def build_sad_cost_volume(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int,
    radius: int,
) -> np.ndarray:
    """Student TODO: return an array with shape H x W x max_disparity."""
    raise NotImplementedError("Implement this in the notebook.")


def winner_takes_all(cost_volume: np.ndarray) -> np.ndarray:
    """Student TODO: choose the disparity with the lowest matching cost."""
    raise NotImplementedError("Implement this in the notebook.")


def reference_sad_cost_volume(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: int,
    radius: int,
) -> np.ndarray:
    """Reference implementation used by instructor cells and quick checks."""
    cv2 = _cv2()
    h, w = left_gray.shape
    volume = np.full((h, w, max_disparity), np.inf, dtype=np.float32)
    kernel_size = 2 * radius + 1
    for d in range(max_disparity):
        shifted = np.zeros_like(right_gray)
        if d == 0:
            shifted[:, :] = right_gray
        else:
            shifted[:, d:] = right_gray[:, :-d]
        diff = np.abs(left_gray - shifted)
        cost = cv2.boxFilter(
            diff,
            ddepth=-1,
            ksize=(kernel_size, kernel_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        volume[:, d:, d] = cost[:, d:]
    return volume


def reference_winner_takes_all(cost_volume: np.ndarray) -> np.ndarray:
    disparity = np.argmin(cost_volume, axis=-1).astype(np.float32)
    invalid = ~np.isfinite(np.min(cost_volume, axis=-1))
    disparity[invalid] = 0
    return disparity


def opencv_sgbm(left_gray: np.ndarray, right_gray: np.ndarray, max_disparity: int = 128) -> np.ndarray:
    cv2 = _cv2()
    max_disparity = int(np.ceil(max_disparity / 16) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(
        (left_gray * 255).astype(np.uint8),
        (right_gray * 255).astype(np.uint8),
    ).astype(np.float32)
    return disp / 16.0
