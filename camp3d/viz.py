from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def show_stereo_pair(left: np.ndarray, right: np.ndarray, figsize=(12, 5)) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(left)
    axes[0].set_title("Left")
    axes[1].imshow(right)
    axes[1].set_title("Right")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()


def show_disparity(disparity: np.ndarray, title: str = "Disparity", vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    valid = np.isfinite(disparity) & (disparity > 0)
    if vmax is None and np.any(valid):
        vmax = float(np.percentile(disparity[valid], 98))
    plt.figure(figsize=(8, 6))
    plt.imshow(disparity, cmap="magma", vmin=0, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(label="pixels")
    plt.tight_layout()


def show_render_comparison(
    target: np.ndarray,
    render: np.ndarray,
    alpha: np.ndarray | None = None,
    metrics: dict[str, float] | None = None,
    figsize=(14, 4),
) -> None:
    """Show target, rendered image, absolute error, and optionally alpha."""
    import matplotlib.pyplot as plt

    target = np.asarray(target, dtype=np.float32).clip(0.0, 1.0)
    render = np.asarray(render, dtype=np.float32).clip(0.0, 1.0)
    error = np.mean(np.abs(target - render), axis=-1)
    titles = ["target", "3DGS render", "absolute error"]
    images = [target, render, error]
    cmaps = [None, None, "magma"]
    if alpha is not None:
        titles.append("alpha")
        images.append(alpha)
        cmaps.append("gray")

    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, image, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    if metrics:
        text = " | ".join(f"{name.upper()}: {value:.3f}" for name, value in metrics.items())
        fig.suptitle(text, y=1.02)
    plt.tight_layout()
    try:
        from IPython.display import display
    except ModuleNotFoundError:
        plt.show()
    else:
        display(fig)
        plt.close(fig)


def save_render_gif(path: str | Path, frames: list[np.ndarray], fps: int = 8) -> Path:
    """Save normalized RGB frames as a GIF for notebook display."""
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [
        np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
        for frame in frames
    ]
    iio.imwrite(path, rgb_frames, duration=1000 / fps, loop=0)
    return path


def _axis_equal_3d(ax, values: np.ndarray) -> None:
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.nanmax(maxs - mins) / 2.0)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def show_colmap_reconstruction(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    camtoworlds: np.ndarray | list[np.ndarray] | None = None,
    image_names: list[str] | None = None,
    max_points: int = 50_000,
    point_size: float = 1.0,
    camera_scale: float | None = None,
    figsize=(9, 8),
    elev: float = 25,
    azim: float = -70,
    normalize: bool = False,
):
    """Show sparse COLMAP points and camera poses in a notebook cell."""
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        raise ValueError("No sparse points to visualize.")

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.size == 0:
        raise ValueError("No finite sparse points to visualize.")
    if colors is not None:
        colors = np.asarray(colors)[valid]

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max(initial=0) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    if normalize:
        points, camtoworlds = _normalize_points_and_cameras(points, camtoworlds)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors if colors is not None else "0.3",
        s=point_size,
        alpha=0.8,
        linewidths=0,
    )

    bounds = [points]
    if camtoworlds is not None:
        camtoworlds = np.asarray(camtoworlds, dtype=np.float32)
        centers = camtoworlds[:, :3, 3]
        bounds.append(centers)
        if camera_scale is None:
            scene_extent = np.linalg.norm(np.nanpercentile(points, 95, axis=0) - np.nanpercentile(points, 5, axis=0))
            camera_scale = float(scene_extent / 25.0) if np.isfinite(scene_extent) and scene_extent > 0 else 0.1

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="#d62728", s=24, depthshade=False, label="cameras")
        for idx, camtoworld in enumerate(camtoworlds):
            center = camtoworld[:3, 3]
            right = camtoworld[:3, 0] * camera_scale * 0.6
            up = -camtoworld[:3, 1] * camera_scale * 0.4
            forward = camtoworld[:3, 2] * camera_scale
            corners = np.stack(
                [
                    center + forward - right - up,
                    center + forward + right - up,
                    center + forward + right + up,
                    center + forward - right + up,
                ]
            )
            for corner in corners:
                ax.plot(*zip(center, corner), color="#d62728", linewidth=0.8, alpha=0.7)
            loop = np.vstack([corners, corners[0]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#d62728", linewidth=0.8, alpha=0.7)
            ax.quiver(
                center[0],
                center[1],
                center[2],
                forward[0],
                forward[1],
                forward[2],
                color="#1f77b4",
                length=1.0,
                normalize=False,
                linewidth=0.8,
            )
            if image_names and idx < len(image_names):
                ax.text(center[0], center[1], center[2], Path(image_names[idx]).stem, fontsize=6, color="#8c1515")

    _axis_equal_3d(ax, np.concatenate(bounds, axis=0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("COLMAP sparse points and camera poses")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if camtoworlds is not None:
        ax.legend(loc="upper right")
    plt.tight_layout()
    return fig, ax


def _prepare_colmap_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    max_points: int = 50_000,
) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        raise ValueError("No sparse points to visualize.")

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.size == 0:
        raise ValueError("No finite sparse points to visualize.")
    if colors is not None:
        colors = np.asarray(colors)[valid]

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max(initial=0) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    return points, colors


def _normalize_points_and_cameras(
    points: np.ndarray,
    camtoworlds: np.ndarray | list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float32)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.nanmax(maxs - mins))
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0
    points = (points - center) / extent + 0.5

    if camtoworlds is None:
        return points, None
    camtoworlds = np.asarray(camtoworlds, dtype=np.float32).copy()
    camtoworlds[:, :3, 3] = (camtoworlds[:, :3, 3] - center) / extent + 0.5
    return points, camtoworlds


def _rgb_strings(colors: np.ndarray) -> list[str]:
    colors_255 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in colors_255]


def show_colmap_reconstruction_plotly(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    camtoworlds: np.ndarray | list[np.ndarray] | None = None,
    image_names: list[str] | None = None,
    max_points: int = 50_000,
    point_size: float = 2.0,
    camera_scale: float | None = None,
    height: int = 720,
    title: str = "COLMAP sparse points and camera poses",
    normalize: bool = False,
):
    """Show sparse COLMAP points and camera poses with Plotly WebGL."""
    import plotly.graph_objects as go

    points, colors = _prepare_colmap_points(points, colors, max_points=max_points)
    if normalize:
        points, camtoworlds = _normalize_points_and_cameras(points, camtoworlds)
    marker = {"size": point_size, "opacity": 0.85}
    if colors is not None:
        marker["color"] = _rgb_strings(colors)
    else:
        marker["color"] = "rgba(70,70,70,0.85)"

    traces = [
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=marker,
            name="sparse points",
            hoverinfo="skip",
        )
    ]

    bounds = [points]
    if camtoworlds is not None:
        camtoworlds = np.asarray(camtoworlds, dtype=np.float32)
        centers = camtoworlds[:, :3, 3]
        bounds.append(centers)
        if camera_scale is None:
            scene_extent = np.linalg.norm(np.nanpercentile(points, 95, axis=0) - np.nanpercentile(points, 5, axis=0))
            camera_scale = float(scene_extent / 25.0) if np.isfinite(scene_extent) and scene_extent > 0 else 0.1

        hover = image_names if image_names is not None else [f"camera {idx}" for idx in range(len(centers))]
        traces.append(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                marker={"size": 4, "color": "#d62728"},
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name="cameras",
            )
        )

        line_x: list[float | None] = []
        line_y: list[float | None] = []
        line_z: list[float | None] = []
        dir_x: list[float | None] = []
        dir_y: list[float | None] = []
        dir_z: list[float | None] = []
        for camtoworld in camtoworlds:
            center = camtoworld[:3, 3]
            right = camtoworld[:3, 0] * camera_scale * 0.6
            up = -camtoworld[:3, 1] * camera_scale * 0.4
            forward = camtoworld[:3, 2] * camera_scale
            corners = np.stack(
                [
                    center + forward - right - up,
                    center + forward + right - up,
                    center + forward + right + up,
                    center + forward - right + up,
                ]
            )
            segments = [(center, corner) for corner in corners]
            segments.extend((corners[idx], corners[(idx + 1) % 4]) for idx in range(4))
            for start, end in segments:
                line_x.extend([start[0], end[0], None])
                line_y.extend([start[1], end[1], None])
                line_z.extend([start[2], end[2], None])
            dir_x.extend([center[0], center[0] + forward[0], None])
            dir_y.extend([center[1], center[1] + forward[1], None])
            dir_z.extend([center[2], center[2] + forward[2], None])

        traces.append(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": "#d62728", "width": 2},
                name="camera frustums",
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=dir_x,
                y=dir_y,
                z=dir_z,
                mode="lines",
                line={"color": "#1f77b4", "width": 3},
                name="view directions",
                hoverinfo="skip",
            )
        )

    all_values = np.concatenate(bounds, axis=0)
    mins = np.nanmin(all_values, axis=0)
    maxs = np.nanmax(all_values, axis=0)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis": {"title": "x", "range": [mins[0], maxs[0]]},
            "yaxis": {"title": "y", "range": [mins[1], maxs[1]]},
            "zaxis": {"title": "z", "range": [mins[2], maxs[2]]},
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
    )
    return fig


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
