# CAMP II: 3D Reconstruction in Endoscopy

This repository contains expanded exercise notebooks for the Computer Aided Medical Procedures II lecture on 3D reconstruction.

The exercise uses a small 640 px wide SCARED stereo-endoscopy subset in `scared_640/`. The same frames are used across all notebooks so students can compare classical structure-from-motion, learned multi-view depth, and stereo reconstruction on the same anatomy.

## Notebooks

For Colab, use `notebooks/colmap_vs_da3_and_3dgs.ipynb`. It combines the COLMAP and DA3 workflows in one runtime, so both pipelines can share the same temporary output root without Google Drive.

1. `notebooks/colmap_vs_da3_and_3dgs.ipynb`
   - Load bundled COLMAP sparse reconstructions for `all_frames`, `every_5th`, and `every_10th`.
   - Inspect the recovered cameras and sparse reconstructions without recomputing COLMAP in Colab.
   - Instantiate a small Gaussian Splatting model directly with PyTorch and `gsplat`.
   - Run DA3-Streaming on the same frame sequence in the same notebook runtime.
   - Backproject streamed DA3 depth into initial Gaussians and compare against the COLMAP-based result.
   - Discuss why endoscopic reconstruction is difficult for feature-based and learned pipelines.

2. `notebooks/03_stereo_reconstruction_expanded.ipynb`
   - Rectify the stereo pairs using the provided endoscope calibration.
   - Implement simple block-matching stereo functions.
   - Compare the student implementation with OpenCV SGBM.
   - Instantiate FoundationStereo directly as a PyTorch model.

## Dataset Layout

```text
scared_640/
  left_image/frame_00001.jpg
  right_image/frame_00001.jpg
  ...
  calibration.json
  endoscope_calibration.yaml
  EXTRACTION_INFO.txt
```

`calibration.json` contains the 640 px intrinsics, distortion coefficients, stereo rotation, and stereo translation used by the notebooks.

## Setup

Create the environment and install the reconstruction stack locally:

```bash
conda env create -f environment.yml
conda activate camp2-3drecon
bash scripts/install_reconstruction_stack.sh
python -m ipykernel install --user --name camp2-3drecon --display-name "CAMP II 3D Reconstruction"
```

For Google Colab, open `notebooks/colmap_vs_da3_and_3dgs.ipynb` and run the first setup cell. It clones this repository into `/content/camp2_3drecon` and changes the notebook runtime into that checkout:

```python
REPO_URL = "https://github.com/FelTris/camp2_3drecon.git"
```

The next notebook cell installs the reconstruction stack from the cloned checkout:

```python
INSTALL_DEPENDENCIES = running_in_colab()
```

The install cell streams the shell script output into the notebook with step markers and timestamps, so if Colab stalls you can see whether it is installing pip packages, installing `gsplat`, or cloning an external repository.

For `gsplat`, the installer first tries the official precompiled wheel index for matching Linux/Python/PyTorch/CUDA combinations. If no matching wheel is available for the current Colab runtime, it falls back to the PyPI package instead of building from a Git checkout.

The installer reuses Colab's existing PyTorch when it is already available. Set `INSTALL_TORCH=1` before running the script if you want to force installation from `requirements-torch-cu118.txt`.

The notebooks use direct Python APIs rather than shelling out to trainer scripts:

- pycolmap / COLMAP: https://colmap.github.io/
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- gsplat: https://github.com/nerfstudio-project/gsplat
- FoundationStereo: https://github.com/NVlabs/FoundationStereo

The FoundationStereo `11-33-40` checkpoint is expected at:

```text
externals/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth
externals/FoundationStereo/pretrained_models/11-33-40/cfg.yaml
```

The notebooks are meant to execute the actual pipeline: model construction, dataloading, forward passes, losses, and optimizer steps are visible in the cells.

## Precomputed COLMAP References

The combined COLMAP + DA3 notebook does not recompute COLMAP by default. Instead, it loads lightweight sparse reconstructions from:

```text
precomputed/01_colmap/all_frames/sparse/
precomputed/01_colmap/every_5th/sparse/
precomputed/01_colmap/every_10th/sparse/
```

Set `RUN_COLMAP_RECONSTRUCTION = True` inside the notebook only if you intentionally want to rerun COLMAP. With this flag enabled, the notebook ignores the bundled sparse models, clears the per-experiment runtime workspace under `outputs/01_colmap/`, and rebuilds feature extraction, sequential matching, and mapping from scratch. Recomputing COLMAP in Colab can take a long time, especially for the full frame sequence.

## Repository Contents

The online repository keeps the exercise notebooks, the `camp3d` helper package, setup scripts, dependency files, tests, the small `scared_640/` sample dataset, and the lightweight precomputed COLMAP sparse references. Local generated outputs, vendored external repositories, full-resolution SCARED data, and temporary test datasets are intentionally ignored.
