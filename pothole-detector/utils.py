"""
utils.py — Shared utility functions for I/O, resizing, annotation parsing, and logging.

Used across all pipeline modules.  Keeps file-handling and evaluation helpers
out of the core CV logic.
"""

import csv
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(log_path: Optional[str] = None) -> logging.Logger:
    """Configure root logger: console (INFO) + optional file (DEBUG).

    Returns the configured logger instance.
    """
    logger = logging.getLogger("pothole_detector")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — DEBUG and above (captures everything)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── Image I/O ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_image(path: str) -> np.ndarray:
    """Read an image from *path* and return it as a BGR NumPy array.

    Raises
    ------
    FileNotFoundError  – if *path* does not exist.
    ValueError         – if the file exists but OpenCV cannot decode it
                         (corrupt / unsupported format).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image (corrupt or unsupported): {path}")

    return img


def resize_image(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale *img* so its longest edge ≤ *max_dim*, preserving aspect ratio.

    If the image is already within bounds, return it unchanged.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img

    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def list_image_files(directory: str) -> List[str]:
    """Return sorted list of absolute paths for supported image files in *directory*."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Input directory not found: {directory}")

    files = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(directory, fname))
    return files


# ─── Annotation Parsing ────────────────────────────────────────────────────

def parse_annotation(xml_path: str) -> List[Tuple[int, int, int, int]]:
    """Parse a Pascal VOC XML annotation file.

    Returns a list of bounding boxes as (xmin, ymin, xmax, ymax) tuples.
    Returns an empty list if the file doesn't exist or contains no objects.
    """
    if not os.path.isfile(xml_path):
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = int(bndbox.findtext("xmin", "0"))
            ymin = int(bndbox.findtext("ymin", "0"))
            xmax = int(bndbox.findtext("xmax", "0"))
            ymax = int(bndbox.findtext("ymax", "0"))
            boxes.append((xmin, ymin, xmax, ymax))
        except ValueError:
            continue

    return boxes


def get_annotation_path(image_path: str, annotation_dir: str) -> str:
    """Derive the expected XML annotation path for a given image path."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(annotation_dir, basename + ".xml")


# ─── IoU Evaluation ────────────────────────────────────────────────────────

def compute_iou(box_a: Tuple[int, int, int, int],
                box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two (xmin, ymin, xmax, ymax) boxes.

    Returns 0.0 if there is no overlap.
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def mask_to_bboxes(mask: np.ndarray,
                   min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    """Extract bounding boxes from a binary mask via contour detection.

    Filters out contours smaller than *min_area* pixels.

    NOTE: This introduces an inherent IoU penalty for organic-shaped pothole
    masks vs. ground-truth bounding boxes.  A tight contour-derived box will
    always under-fill the GT box even for "correct" segmentations.  Worth
    noting in the report's "Challenges Faced" section.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes


# ─── Directory Helpers ──────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create directories (and parents) if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
