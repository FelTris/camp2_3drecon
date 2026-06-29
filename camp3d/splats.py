from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install PyTorch before running the 3DGS notebooks.") from exc
    return torch


def _gsplat_rasterization():
    try:
        from gsplat.rendering import rasterization
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install gsplat before running the 3DGS notebooks.") from exc
    return rasterization


def _to_tensor_image(path: str | Path, scale: float = 1.0):
    torch = _torch()
    image = Image.open(path).convert("RGB")
    if scale != 1.0:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)


@dataclass
class PosedImage:
    image_path: Path
    camtoworld: np.ndarray
    K: np.ndarray


class PosedImageDataset:
    def __init__(self, frames: list[PosedImage], image_scale: float = 0.25):
        self.frames = frames
        self.image_scale = image_scale

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        torch = _torch()
        frame = self.frames[idx]
        image = _to_tensor_image(frame.image_path, self.image_scale)
        K = torch.from_numpy(frame.K.astype(np.float32)).clone()
        K[:2] *= self.image_scale
        return {
            "image": image,
            "camtoworld": torch.from_numpy(frame.camtoworld.astype(np.float32)),
            "K": K,
            "image_path": str(frame.image_path),
        }


class GaussianSplatModel:
    """Small pedagogical 3DGS module around gsplat's rasterizer."""

    def __init__(self, points, colors, init_scale: float = 0.01, init_opacity: float = 0.1):
        torch = _torch()
        self.torch = torch
        self.nn = torch.nn

        points = torch.as_tensor(points, dtype=torch.float32)
        init_colors = torch.as_tensor(colors, dtype=torch.float32).clamp(0, 1)
        n = points.shape[0]

        class _Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.means = torch.nn.Parameter(points)
                self.log_scales = torch.nn.Parameter(
                    torch.full((n, 3), float(np.log(init_scale)), dtype=torch.float32)
                )
                quats = torch.zeros((n, 4), dtype=torch.float32)
                quats[:, 0] = 1.0
                self.quats = torch.nn.Parameter(quats)
                opacity_logit = np.log(init_opacity / (1.0 - init_opacity))
                self.opacity_logits = torch.nn.Parameter(
                    torch.full((n,), float(opacity_logit), dtype=torch.float32)
                )
                color_values = init_colors.clamp(1e-4, 1 - 1e-4)
                self.color_logits = torch.nn.Parameter(torch.logit(color_values))

            def forward(self, camtoworld, K, height: int, width: int):
                import torch.nn.functional as F

                rasterization = _gsplat_rasterization()
                camtoworld = camtoworld.unsqueeze(0) if camtoworld.ndim == 2 else camtoworld
                K = K.unsqueeze(0) if K.ndim == 2 else K
                renders, alphas, info = rasterization(
                    means=self.means,
                    quats=F.normalize(self.quats, dim=-1),
                    scales=torch.exp(self.log_scales),
                    opacities=torch.sigmoid(self.opacity_logits),
                    colors=torch.sigmoid(self.color_logits),
                    viewmats=torch.linalg.inv(camtoworld),
                    Ks=K,
                    width=width,
                    height=height,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1000.0,
                )
                return renders, alphas, info

        self.module = _Module()

    def to(self, device):
        self.module = self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def make_gaussian_optimizers(model: GaussianSplatModel, lr: float = 1e-2):
    torch = _torch()
    module = model.module
    return torch.optim.Adam(
        [
            {"params": [module.means], "lr": lr * 0.1},
            {"params": [module.log_scales], "lr": lr},
            {"params": [module.quats], "lr": lr},
            {"params": [module.opacity_logits], "lr": lr},
            {"params": [module.color_logits], "lr": lr},
        ],
        eps=1e-8,
    )


def _scaled_intrinsics(K: np.ndarray, image_scale: float):
    torch = _torch()
    K_scaled = torch.from_numpy(np.asarray(K, dtype=np.float32)).clone()
    K_scaled[:2] *= image_scale
    return K_scaled


def render_camera(
    model: GaussianSplatModel,
    camtoworld: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Render a camera without loading a target image."""
    torch = _torch()
    model.to(device)
    camtoworld_tensor = torch.from_numpy(np.asarray(camtoworld, dtype=np.float32)).to(device)
    K_tensor = torch.from_numpy(np.asarray(K, dtype=np.float32)).to(device)
    with torch.no_grad():
        render, alpha, _ = model(camtoworld_tensor, K_tensor, height=height, width=width)
    return {
        "render": render[0, ..., :3].detach().cpu().numpy().clip(0.0, 1.0),
        "alpha": alpha[0, ..., 0].detach().cpu().numpy().clip(0.0, 1.0),
    }


def render_posed_image(
    model: GaussianSplatModel,
    frame: PosedImage,
    image_scale: float = 0.25,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Render one posed image and return numpy arrays for notebook inspection."""
    torch = _torch()
    model.to(device)
    image = _to_tensor_image(frame.image_path, image_scale).to(device)
    K = _scaled_intrinsics(frame.K, image_scale).to(device)
    height, width = image.shape[:2]
    rendered = render_camera(model, frame.camtoworld, K.cpu().numpy(), height=height, width=width, device=device)
    return {
        "target": image.detach().cpu().numpy(),
        "render": rendered["render"],
        "alpha": rendered["alpha"],
    }


def render_posed_sequence(
    model: GaussianSplatModel,
    frames: list[PosedImage],
    image_scale: float = 0.25,
    device: str = "cuda",
) -> list[dict[str, np.ndarray]]:
    """Render a list of posed images using the same model."""
    return [render_posed_image(model, frame, image_scale=image_scale, device=device) for frame in frames]


def render_virtual_camera(
    model: GaussianSplatModel,
    camtoworld: np.ndarray,
    K: np.ndarray,
    reference_frame: PosedImage,
    image_scale: float = 0.25,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Render a novel camera using a reference frame only for image dimensions."""
    image = Image.open(reference_frame.image_path)
    height = round(image.height * image_scale)
    width = round(image.width * image_scale)
    K_scaled = _scaled_intrinsics(K, image_scale).numpy()
    return render_camera(model, camtoworld, K_scaled, height=height, width=width, device=device)


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = q0 + t * (q1 - q0)
        return quat / np.linalg.norm(quat)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_camera_path(
    anchor_frames: list[PosedImage],
    steps_per_segment: int = 20,
    include_keyframes: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Interpolate camera poses between anchors using linear translation and rotation SLERP."""
    if len(anchor_frames) < 2:
        raise ValueError("Need at least two anchor frames to interpolate a camera path.")
    path = []
    for start, end in zip(anchor_frames[:-1], anchor_frames[1:]):
        if include_keyframes:
            ts = np.linspace(0.0, 1.0, steps_per_segment, endpoint=False)
        else:
            ts = np.linspace(0.0, 1.0, steps_per_segment + 2)[1:-1]
        q0 = _rotation_matrix_to_quaternion(start.camtoworld[:3, :3])
        q1 = _rotation_matrix_to_quaternion(end.camtoworld[:3, :3])
        for t in ts:
            camtoworld = np.eye(4, dtype=np.float32)
            camtoworld[:3, :3] = _quaternion_to_rotation_matrix(_slerp_quaternion(q0, q1, float(t)))
            camtoworld[:3, 3] = (1.0 - t) * start.camtoworld[:3, 3] + t * end.camtoworld[:3, 3]
            K = (1.0 - t) * start.K + t * end.K
            path.append({"camtoworld": camtoworld, "K": K.astype(np.float32)})
    if include_keyframes:
        last = anchor_frames[-1]
        path.append({"camtoworld": last.camtoworld.astype(np.float32), "K": last.K.astype(np.float32)})
    return path


def train_one_epoch(model, dataloader, optimizer, device: str = "cuda") -> float:
    torch = _torch()
    model.to(device)
    losses = []
    for batch in dataloader:
        image = batch["image"].to(device)
        camtoworld = batch["camtoworld"].to(device)
        K = batch["K"].to(device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        height, width = image.shape[1:3]
        render, _, _ = model(camtoworld, K, height=height, width=width)
        loss = torch.nn.functional.l1_loss(render[..., :3], image)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def points_from_depth_prediction(
    images: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    confidences: np.ndarray | None = None,
    confidence_percentile: float = 25.0,
    stride: int = 8,
    max_points: int = 100_000,
):
    """Convert DA3-style depth, intrinsics, and OpenCV/Colmap w2c extrinsics to points."""
    points_all = []
    colors_all = []
    for image, depth, K, w2c, conf in zip(
        images,
        depths,
        intrinsics,
        extrinsics,
        confidences if confidences is not None else [None] * len(images),
    ):
        if w2c.shape == (3, 4):
            w2c4 = np.eye(4, dtype=np.float32)
            w2c4[:3, :4] = w2c
        else:
            w2c4 = w2c.astype(np.float32)
        c2w = np.linalg.inv(w2c4)
        ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
        z = depth[ys, xs].astype(np.float32)
        valid = np.isfinite(z) & (z > 0)
        if conf is not None:
            valid &= conf[ys, xs] > np.percentile(conf, confidence_percentile)
        xs_valid = xs[valid].astype(np.float32)
        ys_valid = ys[valid].astype(np.float32)
        z_valid = z[valid]
        x = (xs_valid - K[0, 2]) * z_valid / K[0, 0]
        y = (ys_valid - K[1, 2]) * z_valid / K[1, 1]
        camera_points = np.stack([x, y, z_valid, np.ones_like(z_valid)], axis=1)
        world_points = (c2w @ camera_points.T).T[:, :3]
        colors = image[ys_valid.astype(np.int32), xs_valid.astype(np.int32), :3] / 255.0
        points_all.append(world_points)
        colors_all.append(colors)

    points = np.concatenate(points_all, axis=0).astype(np.float32)
    colors = np.concatenate(colors_all, axis=0).astype(np.float32)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points, colors
