"""
test_segmentation.py — Unit tests for the segmentation module.

Tests cover:
  - Output validity (binary mask, correct shape, dtype)
  - Threshold method on synthetic images
  - Graph-Cut method on synthetic images
  - Known-shape detection (black circle on white background)
"""

import numpy as np
import pytest
import cv2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PipelineConfig
from segmentation import Segmenter


@pytest.fixture
def simple_gray():
    """A uniform gray image — should produce minimal/no damage mask."""
    return np.full((200, 200), 180, dtype=np.uint8)


@pytest.fixture
def circle_image():
    """White background with a dark circle — the circle should be detected as damage."""
    img = np.full((200, 200), 220, dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, 50, thickness=-1)
    return img


class TestThresholdSegmentation:
    def test_output_is_binary(self, simple_gray):
        seg = Segmenter(PipelineConfig())
        mask = seg.segment(simple_gray, method="threshold")
        unique = np.unique(mask)
        assert all(v in [0, 255] for v in unique)

    def test_output_shape_matches(self, simple_gray):
        seg = Segmenter(PipelineConfig())
        mask = seg.segment(simple_gray, method="threshold")
        assert mask.shape == simple_gray.shape

    def test_dtype_uint8(self, simple_gray):
        seg = Segmenter(PipelineConfig())
        mask = seg.segment(simple_gray, method="threshold")
        assert mask.dtype == np.uint8

    def test_detects_dark_circle(self, circle_image):
        """A dark circle on a bright background should be detected as damage."""
        seg = Segmenter(PipelineConfig(min_contour_area=50))
        mask = seg.segment(circle_image, method="threshold")
        # The circle center region should be marked as damage
        assert mask[100, 100] == 255

    def test_uniform_image_low_damage(self, simple_gray):
        """A uniform image should have very little detected damage."""
        seg = Segmenter(PipelineConfig())
        mask = seg.segment(simple_gray, method="threshold")
        ratio = np.count_nonzero(mask) / mask.size
        assert ratio < 0.1  # less than 10% (should be near 0)


class TestGraphCutSegmentation:
    def test_output_is_binary(self, simple_gray):
        cfg = PipelineConfig(graphcut_resolution=100)
        seg = Segmenter(cfg)
        mask = seg.segment(simple_gray, method="graphcut")
        unique = np.unique(mask)
        assert all(v in [0, 255] for v in unique)

    def test_output_shape_matches(self, simple_gray):
        cfg = PipelineConfig(graphcut_resolution=100)
        seg = Segmenter(cfg)
        mask = seg.segment(simple_gray, method="graphcut")
        assert mask.shape == simple_gray.shape

    def test_detects_dark_circle(self, circle_image):
        """Graph-Cut should also segment the dark circle as damage."""
        cfg = PipelineConfig(graphcut_resolution=100, min_contour_area=20)
        seg = Segmenter(cfg)
        mask = seg.segment(circle_image, method="graphcut")
        # At least some of the circle region should be detected
        circle_region = mask[70:130, 70:130]
        damage_ratio = np.count_nonzero(circle_region) / circle_region.size
        assert damage_ratio > 0.3  # at least 30% of the circle region


class TestMethodSwapping:
    def test_both_methods_produce_valid_masks(self, circle_image):
        """Both methods should produce valid binary masks without errors."""
        for method in ["threshold", "graphcut"]:
            cfg = PipelineConfig(graphcut_resolution=100)
            seg = Segmenter(cfg)
            mask = seg.segment(circle_image, method=method)
            assert mask.shape == circle_image.shape
            assert mask.dtype == np.uint8
            unique = np.unique(mask)
            assert all(v in [0, 255] for v in unique)
