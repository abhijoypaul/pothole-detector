"""
main.py — CLI entry point for the Pothole / Road Damage Detector pipeline.

Usage examples:
    python main.py
    python main.py --method graphcut --max-images 10
    python main.py --input ../images --output results/ --method threshold --save-masks
    python main.py --method graphcut --annotations ../annotations --save-masks
"""

import argparse
import os
import sys
import time
from typing import Dict, List

from config import PipelineConfig
from preprocessing import Preprocessor
from segmentation import Segmenter
from analysis import Analyzer
from utils import (
    ensure_dirs,
    get_annotation_path,
    list_image_files,
    load_image,
    resize_image,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and return namespace."""
    parser = argparse.ArgumentParser(
        description="Pothole / Road Damage Detector — Classical CV Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                                  # default: threshold on ../images\n"
            "  python main.py --method graphcut --max-images 10 # graph-cut on first 10 images\n"
            "  python main.py --save-masks                      # save mask overlays to results/masks/\n"
        ),
    )
    parser.add_argument(
        "--input", default=None,
        help="Input image directory (default: ../images)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for reports and masks (default: results/)",
    )
    parser.add_argument(
        "--method", choices=["threshold", "graphcut"], default=None,
        help="Segmentation method (default: threshold)",
    )
    parser.add_argument(
        "--annotations", default=None,
        help="Annotation directory for IoU evaluation (default: ../annotations)",
    )
    parser.add_argument(
        "--save-masks", action="store_true", default=None,
        help="Save mask overlay images to results/masks/",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit number of images to process (default: all)",
    )
    parser.add_argument(
        "--graphcut-resolution", type=int, default=None,
        help="Max resolution for Graph-Cut downscale (default: 400)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig, overriding defaults with CLI args where given."""
    cfg = PipelineConfig()
    if args.input is not None:
        cfg.input_dir = args.input
    if args.output is not None:
        cfg.output_dir = args.output
    if args.method is not None:
        cfg.method = args.method
    if args.annotations is not None:
        cfg.annotation_dir = args.annotations
    if args.save_masks is not None:
        cfg.save_masks = args.save_masks
    if args.max_images is not None:
        cfg.max_images = args.max_images
    if args.graphcut_resolution is not None:
        cfg.graphcut_resolution = args.graphcut_resolution
    return cfg


def process_single_image(
    image_path: str,
    cfg: PipelineConfig,
    preprocessor: Preprocessor,
    segmenter: Segmenter,
    analyzer: Analyzer,
    logger,
) -> Dict:
    """Run the full pipeline on one image.  Returns a result record dict."""
    import cv2  # local import to keep top-level clean for testing

    filename = os.path.basename(image_path)
    t0 = time.perf_counter()

    # ── Load & resize ───────────────────────────────────────────────
    raw_img = load_image(image_path)
    original_shape = raw_img.shape  # before any resize (for annotation scaling)
    img = resize_image(raw_img, cfg.max_resolution)

    # ── Preprocess ──────────────────────────────────────────────────
    gray = preprocessor.preprocess(img)

    # ── Segment ─────────────────────────────────────────────────────
    mask = segmenter.segment(gray)

    # ── Analyse ─────────────────────────────────────────────────────
    area_ratio = analyzer.compute_area(mask)
    severity = analyzer.score_severity(area_ratio)

    # ── Evaluate against ground truth ───────────────────────────────
    ann_path = get_annotation_path(image_path, cfg.annotation_dir)
    eval_result = analyzer.evaluate_detection(mask, ann_path, original_shape)

    elapsed = time.perf_counter() - t0

    # ── Save mask overlay ───────────────────────────────────────────
    if cfg.save_masks:
        overlay = analyzer.create_overlay(img, mask)
        mask_path = os.path.join(cfg.masks_dir, filename)
        cv2.imwrite(mask_path, overlay)

    record = {
        "filename": filename,
        "damage_area_ratio": round(area_ratio, 6),
        "severity": severity,
        "num_regions_detected": eval_result["det_boxes"],
        "gt_boxes": eval_result["gt_boxes"],
        "best_iou": eval_result["best_iou"],
        "mean_iou": eval_result["mean_iou"],
        "processing_time_sec": round(elapsed, 3),
        "method": cfg.method,
    }
    return record


def print_summary(records: List[Dict], total_time: float) -> None:
    """Print a batch summary to console."""
    n = len(records)
    if n == 0:
        print("\nNo images processed.")
        return

    severities = {"Low": 0, "Medium": 0, "High": 0}
    total_iou = 0.0
    iou_count = 0
    for r in records:
        severities[r["severity"]] = severities.get(r["severity"], 0) + 1
        if r["gt_boxes"] > 0:
            total_iou += r["mean_iou"]
            iou_count += 1

    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    avg_time = total_time / n

    print("\n" + "=" * 60)
    print("  BATCH SUMMARY")
    print("=" * 60)
    print(f"  Images processed : {n}")
    print(f"  Method           : {records[0]['method']}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Avg time/image   : {avg_time:.2f}s")
    print(f"  Severity dist    : Low={severities['Low']}  "
          f"Medium={severities['Medium']}  High={severities['High']}")
    if iou_count > 0:
        print(f"  Mean IoU (GT)    : {avg_iou:.4f}  ({iou_count} images with annotations)")
    print("=" * 60)


def main() -> None:
    """Orchestrate the full batch pipeline."""
    args = parse_args()
    cfg = build_config(args)

    # ── Setup ───────────────────────────────────────────────────────
    ensure_dirs(cfg.output_dir)
    if cfg.save_masks:
        ensure_dirs(cfg.masks_dir)

    logger = setup_logging(cfg.log_path)
    logger.info("Pothole Detector started — method=%s", cfg.method)
    logger.info("Input dir : %s", os.path.abspath(cfg.input_dir))
    logger.info("Output dir: %s", os.path.abspath(cfg.output_dir))

    # ── Discover images ─────────────────────────────────────────────
    image_files = list_image_files(cfg.input_dir)
    if cfg.max_images > 0:
        image_files = image_files[: cfg.max_images]
    total = len(image_files)
    logger.info("Found %d images to process", total)

    if total == 0:
        logger.warning("No images found — exiting")
        return

    # ── Instantiate pipeline components ─────────────────────────────
    preprocessor = Preprocessor(cfg)
    segmenter = Segmenter(cfg)
    analyzer = Analyzer(cfg)

    # ── Process batch ───────────────────────────────────────────────
    records: List[Dict] = []
    skipped = 0
    batch_t0 = time.perf_counter()

    for i, img_path in enumerate(image_files, start=1):
        fname = os.path.basename(img_path)
        try:
            record = process_single_image(
                img_path, cfg, preprocessor, segmenter, analyzer, logger
            )
            records.append(record)
            # Progress line
            print(
                f"  [{i:>{len(str(total))}}/{total}] {fname} — "
                f"{record['severity']} ({record['damage_area_ratio']*100:.1f}%) — "
                f"{record['processing_time_sec']:.2f}s"
            )
        except Exception as e:
            skipped += 1
            logger.error("SKIPPED %s — %s: %s", fname, type(e).__name__, e)

    batch_elapsed = time.perf_counter() - batch_t0

    # ── Export reports ──────────────────────────────────────────────
    analyzer.export_csv(records, cfg.report_csv_path)
    analyzer.export_json(records, cfg.report_json_path)

    # ── Summary ─────────────────────────────────────────────────────
    print_summary(records, batch_elapsed)
    if skipped:
        logger.warning("%d image(s) skipped due to errors — see pipeline.log", skipped)

    logger.info("Pipeline complete. Reports at %s", cfg.output_dir)


if __name__ == "__main__":
    main()
