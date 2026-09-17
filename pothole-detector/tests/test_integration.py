"""
test_integration.py — Integration test for the full pipeline.

Runs the pipeline end-to-end on a few real images from the dataset
(if available) or synthetic images, and validates report output.
"""

import json
import os

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PipelineConfig
from preprocessing import Preprocessor
from segmentation import Segmenter
from analysis import Analyzer
from utils import ensure_dirs, load_image, resize_image, get_annotation_path


def _make_synthetic_pothole_image(size=(300, 400)):
    """Create a synthetic road image with a dark pothole region."""
    h, w = size
    img = np.full((h, w, 3), 160, dtype=np.uint8)  # gray road
    # Add some texture
    rng = np.random.RandomState(42)
    noise = rng.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Draw a dark pothole
    cv2.ellipse(img, (200, 150), (60, 40), 0, 0, 360, (40, 40, 40), -1)
    return img


class TestFullPipeline:
    def test_pipeline_on_synthetic_images(self, tmp_path):
        """Create 3 synthetic images, run pipeline, verify report output."""
        # Setup dirs
        input_dir = str(tmp_path / "input")
        output_dir = str(tmp_path / "output")
        ensure_dirs(input_dir, output_dir)

        # Create synthetic images
        for i in range(3):
            img = _make_synthetic_pothole_image()
            cv2.imwrite(os.path.join(input_dir, f"test_{i}.png"), img)

        # Configure & run
        cfg = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            annotation_dir=str(tmp_path / "annotations"),  # empty → no IoU
            method="threshold",
            save_masks=False,
            max_images=0,
        )
        preprocessor = Preprocessor(cfg)
        segmenter = Segmenter(cfg)
        analyzer = Analyzer(cfg)

        records = []
        from utils import list_image_files
        for img_path in list_image_files(input_dir):
            raw = load_image(img_path)
            img = resize_image(raw, cfg.max_resolution)
            gray = preprocessor.preprocess(img)
            mask = segmenter.segment(gray)
            area = analyzer.compute_area(mask)
            severity = analyzer.score_severity(area)
            records.append({
                "filename": os.path.basename(img_path),
                "damage_area_ratio": round(area, 6),
                "severity": severity,
            })

        # Export
        csv_path = os.path.join(output_dir, "report.csv")
        json_path = os.path.join(output_dir, "report.json")
        analyzer.export_csv(records, csv_path)
        analyzer.export_json(records, json_path)

        # Validate
        assert os.path.isfile(csv_path)
        assert os.path.isfile(json_path)

        with open(json_path, "r") as f:
            loaded = json.load(f)
        assert len(loaded) == 3
        for rec in loaded:
            assert "filename" in rec
            assert "severity" in rec
            assert rec["severity"] in ["Low", "Medium", "High"]

    def test_pipeline_graphcut_on_synthetic(self, tmp_path):
        """Verify graph-cut method also runs without errors on synthetic data."""
        input_dir = str(tmp_path / "input")
        output_dir = str(tmp_path / "output")
        ensure_dirs(input_dir, output_dir)

        img = _make_synthetic_pothole_image(size=(100, 100))
        cv2.imwrite(os.path.join(input_dir, "gc_test.png"), img)

        cfg = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            annotation_dir=str(tmp_path / "annotations"),
            method="graphcut",
            graphcut_resolution=50,  # tiny for speed
            save_masks=False,
        )
        preprocessor = Preprocessor(cfg)
        segmenter = Segmenter(cfg)
        analyzer = Analyzer(cfg)

        raw = load_image(os.path.join(input_dir, "gc_test.png"))
        gray = preprocessor.preprocess(raw)
        mask = segmenter.segment(gray)

        assert mask.shape == gray.shape
        assert mask.dtype == np.uint8
        area = analyzer.compute_area(mask)
        assert 0.0 <= area <= 1.0


class TestErrorHandling:
    def test_corrupt_image_raises(self, tmp_path):
        """A non-image file should raise ValueError on load."""
        bad_file = str(tmp_path / "corrupt.png")
        with open(bad_file, "w") as f:
            f.write("this is not an image")

        with pytest.raises(ValueError, match="Failed to decode"):
            load_image(bad_file)

    def test_missing_image_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")
