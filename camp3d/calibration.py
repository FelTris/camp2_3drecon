from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for stereo rectification. Install the notebook "
            "requirements with: python -m pip install -r requirements.txt"
        ) from exc
    return cv2


@dataclass(frozen=True)
class StereoCalibration:
    left_K: np.ndarray
    left_distortion: np.ndarray
    right_K: np.ndarray
    right_distortion: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple[int, int]
    unit: str = "mm"

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T.reshape(-1)))

    @property
    def fx(self) -> float:
        return float(self.left_K[0, 0])


def load_calibration(path: str | Path) -> StereoCalibration:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return StereoCalibration(
        left_K=np.asarray(raw["left"]["K"], dtype=np.float64),
        left_distortion=np.asarray(raw["left"]["distortion"], dtype=np.float64),
        right_K=np.asarray(raw["right"]["K"], dtype=np.float64),
        right_distortion=np.asarray(raw["right"]["distortion"], dtype=np.float64),
        R=np.asarray(raw["stereo"]["R"], dtype=np.float64),
        T=np.asarray(raw["stereo"]["T"], dtype=np.float64),
        image_size=tuple(raw["image_size"]),
        unit=raw.get("unit", "mm"),
    )


def stereo_rectification_maps(calib: StereoCalibration):
    cv2 = _cv2()
    image_size = tuple(int(v) for v in calib.image_size)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib.left_K,
        calib.left_distortion,
        calib.right_K,
        calib.right_distortion,
        image_size,
        calib.R,
        calib.T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib.left_K, calib.left_distortion, R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib.right_K, calib.right_distortion, R2, P2, image_size, cv2.CV_32FC1
    )
    return (map1x, map1y), (map2x, map2y), {"R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q}


def rectify_pair(left_rgb: np.ndarray, right_rgb: np.ndarray, calib: StereoCalibration):
    cv2 = _cv2()
    left_maps, right_maps, rect = stereo_rectification_maps(calib)
    left_rect = cv2.remap(left_rgb, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_rgb, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
    return left_rect, right_rect, rect


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    disparity = disparity.astype(np.float32)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = float(fx) * float(baseline) / disparity[valid]
    return depth


def depth_to_point_cloud(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, stride: int = 4):
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1)
    colors = rgb[ys.astype(np.int32), xs.astype(np.int32), :3].reshape(-1, 3)
    return points, colors
