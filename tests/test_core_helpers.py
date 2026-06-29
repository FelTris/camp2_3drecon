from __future__ import annotations

import numpy as np

from camp3d.calibration import disparity_to_depth, load_calibration
from camp3d.data import ScaredSubset
from camp3d.metrics import image_metrics
from camp3d.stereo import reference_winner_takes_all


def test_scared_subset_layout_matches_sample_data() -> None:
    dataset = ScaredSubset()
    summary = dataset.validate()

    assert summary["num_pairs"] == 191
    assert summary["has_calibration_json"] is True
    assert summary["has_calibration_yaml"] is True


def test_calibration_and_depth_conversion() -> None:
    dataset = ScaredSubset()
    calibration = load_calibration(dataset.calibration_json)
    disparity = np.array([[0.0, 2.0], [4.0, -1.0]], dtype=np.float32)

    depth = disparity_to_depth(disparity, calibration.fx, calibration.baseline)

    assert np.isnan(depth[0, 0])
    assert np.isnan(depth[1, 1])
    assert np.isfinite(depth[0, 1])
    assert depth[0, 1] > depth[1, 0]


def test_metrics_and_winner_takes_all_are_deterministic() -> None:
    target = np.zeros((8, 8, 3), dtype=np.float32)
    pred = target.copy()
    scores = image_metrics(pred, target)

    assert np.isinf(scores["psnr"])
    assert scores["ssim"] == 1.0

    cost_volume = np.array([[[3.0, 1.0, 2.0], [np.inf, np.inf, np.inf]]], dtype=np.float32)
    disparity = reference_winner_takes_all(cost_volume)

    assert disparity[0, 0] == 1.0
    assert disparity[0, 1] == 0.0
