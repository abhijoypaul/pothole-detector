"""
preprocessing.py — Image preprocessing pipeline (FR-1).

Converts raw BGR images to cleaned, contrast-enhanced grayscale images
ready for segmentation.  Each step is a separate method for testability.

Pipeline:  BGR → Grayscale → Denoise → CLAHE contrast enhancement
"""

import cv2
import numpy as np

from config import PipelineConfig


class Preprocessor:
    """Stateless image preprocessor.  All behaviour controlled via *cfg*."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        # Pre-build the CLAHE object (reusable across images)
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=cfg.clahe_grid,
        )

    # ── Individual steps ────────────────────────────────────────────────

    def to_grayscale(self, img_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR image to single-channel grayscale."""
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply blur-based noise reduction.

        Supports Gaussian and median blur, selected via config.
        """
        k = self.cfg.blur_kernel
        if self.cfg.blur_type == "median":
            return cv2.medianBlur(gray, k)
        else:
            # Default to Gaussian
            return cv2.GaussianBlur(gray, (k, k), 0)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        CLAHE prevents over-amplification of noise in homogeneous regions
        while boosting local contrast — important for revealing subtle
        pothole texture differences against the road surface.
        """
        return self._clahe.apply(gray)

    # ── Full pipeline ───────────────────────────────────────────────────

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run the full preprocessing chain.

        Parameters
        ----------
        img_bgr : np.ndarray
            Input BGR image (uint8, shape H×W×3).

        Returns
        -------
        np.ndarray
            Cleaned, contrast-enhanced grayscale image (uint8, shape H×W).
        """
        gray = self.to_grayscale(img_bgr)
        denoised = self.denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        return enhanced
