"""
analysis.py — Damage analysis and reporting (FR-3).

Computes damage area, assigns severity scores, evaluates detection
against ground-truth annotations, and exports batch reports to CSV/JSON.
"""

import csv
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import PipelineConfig
from utils import compute_iou, mask_to_bboxes, parse_annotation

logger = logging.getLogger("pothole_detector")


class Analyzer:
    """Analyse a binary damage mask and produce structured results."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ── Core metrics ────────────────────────────────────────────────────

    @staticmethod
    def compute_area(mask: np.ndarray) -> float:
        """Return the ratio of damage pixels to total pixels (0.0–1.0).

        Parameters
        ----------
        mask : np.ndarray
            Binary mask (uint8) — 255 for damage, 0 for road.
        """
        total = mask.size
        if total == 0:
            return 0.0
        damage = np.count_nonzero(mask)
        return damage / total

    def score_severity(self, area_ratio: float) -> str:
        """Map an area ratio to a severity bucket.

        Thresholds are configurable via config.  After the first batch run,
        inspect the distribution and adjust so all three buckets are
        actually populated.
        """
        if area_ratio < self.cfg.severity_low:
            return "Low"
        elif area_ratio > self.cfg.severity_high:
            return "High"
        else:
            return "Medium"

    # ── Ground-truth evaluation ─────────────────────────────────────────

    def evaluate_detection(
        self,
        mask: np.ndarray,
        annotation_path: str,
        original_shape: Tuple[int, int],
    ) -> Dict:
        """Compare detected mask against ground-truth bounding boxes.

        Returns a dict with:
          - gt_boxes:     number of ground-truth boxes
          - det_boxes:    number of detected bounding regions
          - best_iou:     best IoU between any predicted/GT box pair
          - mean_iou:     mean of best-match IoUs across GT boxes

        NOTE: IoU is computed between mask-derived bounding boxes and GT
        bounding boxes.  Organic mask shapes inherently under-fill boxes,
        so even correct segmentations will show some IoU penalty.
        """
        gt_boxes = parse_annotation(annotation_path)
        if not gt_boxes:
            return {
                "gt_boxes": 0,
                "det_boxes": 0,
                "best_iou": 0.0,
                "mean_iou": 0.0,
            }

        # Scale GT boxes if mask was resized relative to original annotation dims
        # (annotations are in original image coordinates)
        mask_h, mask_w = mask.shape[:2]
        orig_h, orig_w = original_shape[:2]

        if (mask_h != orig_h) or (mask_w != orig_w):
            sx = mask_w / orig_w
            sy = mask_h / orig_h
            gt_boxes = [
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                for (x1, y1, x2, y2) in gt_boxes
            ]

        det_boxes = mask_to_bboxes(mask, min_area=self.cfg.min_contour_area)

        if not det_boxes:
            return {
                "gt_boxes": len(gt_boxes),
                "det_boxes": 0,
                "best_iou": 0.0,
                "mean_iou": 0.0,
            }

        # For each GT box, find the best-matching detection
        best_ious = []
        overall_best = 0.0
        for gt in gt_boxes:
            best = 0.0
            for det in det_boxes:
                iou = compute_iou(gt, det)
                best = max(best, iou)
                overall_best = max(overall_best, iou)
            best_ious.append(best)

        return {
            "gt_boxes": len(gt_boxes),
            "det_boxes": len(det_boxes),
            "best_iou": round(overall_best, 4),
            "mean_iou": round(float(np.mean(best_ious)), 4),
        }

    # ── Mask overlay for visual inspection ──────────────────────────────

    @staticmethod
    def create_overlay(image_bgr: np.ndarray,
                       mask: np.ndarray,
                       alpha: float = 0.4) -> np.ndarray:
        """Overlay damage mask on original image as a semi-transparent red tint."""
        overlay = image_bgr.copy()
        # Resize mask to match image if needed
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        # Red tint on damage areas
        overlay[mask == 255] = [0, 0, 255]
        blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
        return blended

    # ── Report export ───────────────────────────────────────────────────

    @staticmethod
    def export_csv(records: List[Dict], path: str) -> None:
        """Write a list of record dicts to a CSV file."""
        if not records:
            logger.warning("No records to write to CSV")
            return

        fieldnames = list(records[0].keys())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info("CSV report written: %s (%d rows)", path, len(records))

    @staticmethod
    def export_json(records: List[Dict], path: str) -> None:
        """Write a list of record dicts to a pretty-printed JSON file."""
        if not records:
            logger.warning("No records to write to JSON")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info("JSON report written: %s (%d records)", path, len(records))
