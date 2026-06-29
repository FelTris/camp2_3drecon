from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, max_value: float = 1.0) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_value) - 10 * np.log10(mse))


def ssim(pred: np.ndarray, target: np.ndarray, max_value: float = 1.0) -> float:
    """Compute mean structural similarity for images in HxW or HxWxC format."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"SSIM expects matching shapes, got {pred.shape} and {target.shape}.")

    if pred.ndim == 2:
        pred = pred[..., None]
        target = target[..., None]
    if pred.ndim != 3:
        raise ValueError("SSIM expects an image with shape HxW or HxWxC.")

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    window = min(11, pred.shape[0], pred.shape[1])
    if window % 2 == 0:
        window -= 1
    window = max(window, 1)

    values = []
    for channel in range(pred.shape[-1]):
        x = pred[..., channel]
        y = target[..., channel]
        mu_x = _mean_filter(x, window)
        mu_y = _mean_filter(y, window)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = _mean_filter(x * x, window) - mu_x2
        sigma_y2 = _mean_filter(y * y, window) - mu_y2
        sigma_xy = _mean_filter(x * y, window) - mu_xy
        score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        values.append(float(np.mean(score)))
    return float(np.mean(values))


def _mean_filter(image: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return image.astype(np.float32)
    pad = window // 2
    padded = np.pad(image, pad_width=pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return windows.mean(axis=(-2, -1), dtype=np.float32)


def image_metrics(pred: np.ndarray, target: np.ndarray, max_value: float = 1.0) -> dict[str, float]:
    """Return common full-reference image metrics for two normalized images."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return {
        "psnr": psnr(pred, target, max_value=max_value),
        "ssim": ssim(pred, target, max_value=max_value),
    }


def disparity_smoothness(disparity: np.ndarray) -> float:
    valid = np.isfinite(disparity)
    dx = np.abs(np.diff(disparity, axis=1))
    dy = np.abs(np.diff(disparity, axis=0))
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]
    values = []
    if np.any(valid_x):
        values.append(dx[valid_x])
    if np.any(valid_y):
        values.append(dy[valid_y])
    if not values:
        return float("nan")
    return float(np.mean(np.concatenate(values)))
