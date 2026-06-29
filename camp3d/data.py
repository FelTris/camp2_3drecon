from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root from an imported notebook or script."""
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ScaredSubset:
    root: Path | str = get_repo_root() / "scared_640"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def left_dir(self) -> Path:
        return self.root / "left_image"

    @property
    def right_dir(self) -> Path:
        return self.root / "right_image"

    @property
    def calibration_json(self) -> Path:
        return self.root / "calibration.json"

    @property
    def calibration_yaml(self) -> Path:
        return self.root / "endoscope_calibration.yaml"

    @staticmethod
    def _images(directory: Path) -> list[Path]:
        paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(directory.glob(pattern))
        return sorted(paths)

    def left_images(self) -> list[Path]:
        return self._images(self.left_dir)

    def right_images(self) -> list[Path]:
        return self._images(self.right_dir)

    def stereo_pairs(self) -> list[tuple[Path, Path]]:
        left = self.left_images()
        right = self.right_images()
        right_by_stem = {path.stem: path for path in right}
        pairs = [(path, right_by_stem[path.stem]) for path in left if path.stem in right_by_stem]
        if len(pairs) != len(left) or len(pairs) != len(right):
            raise ValueError(
                f"Found {len(left)} left images, {len(right)} right images, "
                f"and {len(pairs)} matching pairs."
            )
        return pairs

    def validate(self) -> dict[str, object]:
        pairs = self.stereo_pairs()
        return {
            "root": str(self.root),
            "num_pairs": len(pairs),
            "first_pair": tuple(str(p) for p in pairs[0]) if pairs else None,
            "has_calibration_json": self.calibration_json.exists(),
            "has_calibration_yaml": self.calibration_yaml.exists(),
        }
