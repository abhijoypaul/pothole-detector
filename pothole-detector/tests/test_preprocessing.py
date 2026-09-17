"""
test_preprocessing.py — Unit tests for the preprocessing module.

Tests cover:
  - Grayscale conversion (output shape, dtype)
  - Denoising (output shape, dtype, both blur types)
  - CLAHE contrast enhancement (output shape, contrast improvement)
  - Full pipeline chain
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PipelineConfig
from preprocessing import Preprocessor


@pytest.fixture
def sample_bgr():
    """Create a synthetic 100×150 BGR image with a gradient + noise."""
    rng = np.random.RandomState(42)
    # Gradient base
    base = np.tile(np.linspace(30, 220, 150, dtype=np.uint8), (100, 1))
    img = np.stack([base, base, base], axis=-1)
    # Add noise
    noise = rng.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def preprocessor():
    return Preprocessor(PipelineConfig())


class TestGrayscale:
    def test_output_is_2d(self, preprocessor, sample_bgr):
        gray = preprocessor.to_grayscale(sample_bgr)
        assert gray.ndim == 2

    def test_shape_matches(self, preprocessor, sample_bgr):
        gray = preprocessor.to_grayscale(sample_bgr)
        assert gray.shape == sample_bgr.shape[:2]

    def test_dtype_uint8(self, preprocessor, sample_bgr):
        gray = preprocessor.to_grayscale(sample_bgr)
        assert gray.dtype == np.uint8


class TestDenoise:
    def test_gaussian_output_shape(self, preprocessor, sample_bgr):
        gray = preprocessor.to_grayscale(sample_bgr)
        denoised = preprocessor.denoise(gray)
        assert denoised.shape == gray.shape
        assert denoised.dtype == np.uint8

    def test_median_output_shape(self, sample_bgr):
        cfg = PipelineConfig(blur_type="median")
        pp = Preprocessor(cfg)
        gray = pp.to_grayscale(sample_bgr)
        denoised = pp.denoise(gray)
        assert denoised.shape == gray.shape
        assert denoised.dtype == np.uint8

    def test_denoise_reduces_noise(self, preprocessor, sample_bgr):
        """Blurring should reduce pixel-to-pixel variation (standard deviation of Laplacian)."""
        gray = preprocessor.to_grayscale(sample_bgr)
        denoised = preprocessor.denoise(gray)
        import cv2
        lap_before = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_after = cv2.Laplacian(denoised, cv2.CV_64F).var()
        assert lap_after <= lap_before


class TestCLAHE:
    def test_output_shape_dtype(self, preprocessor, sample_bgr):
        gray = preprocessor.to_grayscale(sample_bgr)
        enhanced = preprocessor.enhance_contrast(gray)
        assert enhanced.shape == gray.shape
        assert enhanced.dtype == np.uint8

    def test_contrast_improved(self, preprocessor):
        """CLAHE should increase (or at least not decrease) the standard deviation
        of a low-contrast input."""
        low_contrast = np.full((100, 100), 128, dtype=np.uint8)
        # Add very slight variation
        low_contrast[40:60, 40:60] = 120
        enhanced = preprocessor.enhance_contrast(low_contrast)
        assert enhanced.std() >= low_contrast.std()


class TestFullPipeline:
    def test_preprocess_chain(self, preprocessor, sample_bgr):
        result = preprocessor.preprocess(sample_bgr)
        assert result.ndim == 2
        assert result.shape == sample_bgr.shape[:2]
        assert result.dtype == np.uint8
