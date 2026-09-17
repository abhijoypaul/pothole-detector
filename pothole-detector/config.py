"""
config.py — Central configuration for the Pothole/Road Damage Detector pipeline.

All tunable parameters live here. Values can be overridden by CLI flags in main.py.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineConfig:
    """Holds every tunable knob in the pipeline.

    Defaults are calibrated for the Kaggle Pothole Detection Dataset
    (665 PNG images, mixed resolutions).
    """

    # ── Paths ────────────────────────────────────────────────────────────
    input_dir: str = os.path.join("..", "images")
    annotation_dir: str = os.path.join("..", "annotations")
    output_dir: str = "results"

    # ── Segmentation method ──────────────────────────────────────────────
    # "threshold" (adaptive thresholding + morphology)  or
    # "graphcut"  (MRF energy minimization via scipy max-flow)
    method: str = "threshold"

    # ── Resizing ─────────────────────────────────────────────────────────
    # Downscale longest edge to this for ALL processing (preprocessing + threshold)
    max_resolution: int = 800
    # Further downscale for Graph-Cut to keep max-flow tractable.
    # 400 → ~160 000 pixel-nodes; drop to 300/250 if timing exceeds target.
    graphcut_resolution: int = 300

    # ── Preprocessing (FR-1) ─────────────────────────────────────────────
    blur_type: str = "gaussian"       # "gaussian" or "median"
    blur_kernel: int = 5              # must be odd
    clahe_clip: float = 2.0           # CLAHE clip limit
    clahe_grid: Tuple[int, int] = (8, 8)  # CLAHE tile grid size

    # ── Segmentation — Adaptive Threshold (Method A) ─────────────────────
    adaptive_block: int = 31          # block size for adaptive threshold (odd)
    adaptive_c: int = 7              # constant subtracted from mean
    morph_kernel: int = 5             # structuring-element size for open/close
    min_contour_area: int = 100       # discard contours smaller than this (px²)

    # ── Segmentation — Graph-Cut / MRF (Method B) ────────────────────────
    # λ controls the relative weight of the smoothness term vs data term.
    graphcut_lambda: float = 10.0
    # β is auto-computed from image gradients; this is a fallback multiplier.
    graphcut_beta_multiplier: float = 1.0
    # Scaling factor: scipy maximum_flow requires *integer* capacities,
    # so we multiply float costs by this value and cast to int.
    graphcut_capacity_scale: int = 1000

    # ── Severity scoring (FR-3) ──────────────────────────────────────────
    # Area-ratio thresholds.  Tune after first batch run so all three
    # buckets are populated — see plan notes.
    severity_low: float = 0.02        # < 2 % → Low
    severity_high: float = 0.08       # > 8 % → High  (2–8 % → Medium)

    # ── Output options ───────────────────────────────────────────────────
    save_masks: bool = True           # save binary mask + overlay per image
    max_images: int = 0               # 0 = process all images in input_dir

    # ── Derived / convenience ────────────────────────────────────────────
    @property
    def masks_dir(self) -> str:
        return os.path.join(self.output_dir, "masks")

    @property
    def report_csv_path(self) -> str:
        return os.path.join(self.output_dir, "report.csv")

    @property
    def report_json_path(self) -> str:
        return os.path.join(self.output_dir, "report.json")

    @property
    def log_path(self) -> str:
        return os.path.join(self.output_dir, "pipeline.log")
