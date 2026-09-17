"""
test_analysis.py — Unit tests for the analysis module.

Tests cover:
  - Area computation (all-white, all-black, partial)
  - Severity scoring boundary conditions
  - IoU computation with known boxes
  - CSV/JSON export round-trip
"""

import json
import os
import tempfile

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PipelineConfig
from analysis import Analyzer
from utils import compute_iou


@pytest.fixture
def analyzer():
    return Analyzer(PipelineConfig())


class TestComputeArea:
    def test_all_white_mask(self, analyzer):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert analyzer.compute_area(mask) == 1.0

    def test_all_black_mask(self, analyzer):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert analyzer.compute_area(mask) == 0.0

    def test_half_damage(self, analyzer):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 255  # top half is damage
        area = analyzer.compute_area(mask)
        assert abs(area - 0.5) < 0.01

    def test_known_area(self, analyzer):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # 20×20 = 400 pixels out of 40000
        area = analyzer.compute_area(mask)
        assert abs(area - 400 / 40000) < 1e-6

    def test_empty_mask(self, analyzer):
        mask = np.zeros((0,), dtype=np.uint8)
        assert analyzer.compute_area(mask) == 0.0


class TestSeverityScoring:
    def test_low_severity(self, analyzer):
        assert analyzer.score_severity(0.01) == "Low"

    def test_medium_severity(self, analyzer):
        assert analyzer.score_severity(0.05) == "Medium"

    def test_high_severity(self, analyzer):
        assert analyzer.score_severity(0.10) == "High"

    def test_boundary_low_medium(self, analyzer):
        # Exactly at low threshold → should be Medium (>= low, <= high)
        assert analyzer.score_severity(0.02) == "Medium"

    def test_boundary_medium_high(self, analyzer):
        # Exactly at high threshold → should be Medium (<= high)
        assert analyzer.score_severity(0.08) == "Medium"

    def test_just_above_high(self, analyzer):
        assert analyzer.score_severity(0.081) == "High"

    def test_zero_area(self, analyzer):
        assert analyzer.score_severity(0.0) == "Low"


class TestIoU:
    def test_perfect_overlap(self):
        box = (10, 10, 50, 50)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0, 0, 20, 20)
        b = (10, 10, 30, 30)
        # Intersection: 10×10 = 100,  Union: 400 + 400 - 100 = 700
        iou = compute_iou(a, b)
        assert abs(iou - 100 / 700) < 1e-6

    def test_contained_box(self):
        outer = (0, 0, 100, 100)
        inner = (20, 20, 40, 40)
        # Intersection = 20×20 = 400,  Union = 10000 + 400 - 400 = 10000
        iou = compute_iou(outer, inner)
        assert abs(iou - 400 / 10000) < 1e-6


class TestExport:
    def test_csv_round_trip(self, analyzer, tmp_path):
        records = [
            {"filename": "test1.png", "severity": "Low", "area": 0.01},
            {"filename": "test2.png", "severity": "High", "area": 0.15},
        ]
        csv_path = str(tmp_path / "test_report.csv")
        analyzer.export_csv(records, csv_path)
        assert os.path.isfile(csv_path)

        # Verify contents
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["filename"] == "test1.png"

    def test_json_round_trip(self, analyzer, tmp_path):
        records = [
            {"filename": "test1.png", "severity": "Low", "area": 0.01},
        ]
        json_path = str(tmp_path / "test_report.json")
        analyzer.export_json(records, json_path)
        assert os.path.isfile(json_path)

        with open(json_path, "r") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["filename"] == "test1.png"

    def test_empty_records_no_crash(self, analyzer, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        analyzer.export_csv([], csv_path)
        # Should not crash, file may or may not be created
