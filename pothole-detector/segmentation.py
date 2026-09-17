"""
segmentation.py — Damage region segmentation (FR-2).

Two swappable strategies selected via config:
  Method A  "threshold"  — Adaptive thresholding + morphological cleanup
  Method B  "graphcut"   — MRF energy minimization via scipy max-flow

Both produce a binary mask (0 = road, 255 = damage) of the same size as
the input preprocessed grayscale image.
"""

import logging

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from config import PipelineConfig

logger = logging.getLogger("pothole_detector")


class Segmenter:
    """Segment damage regions from a preprocessed grayscale image."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ── Public API ──────────────────────────────────────────────────────

    def segment(self, gray: np.ndarray, method: str = None) -> np.ndarray:
        """Produce a binary damage mask from *gray*.

        Parameters
        ----------
        gray : np.ndarray
            Preprocessed grayscale image (uint8, H×W).
        method : str, optional
            Override the config method ("threshold" or "graphcut").

        Returns
        -------
        np.ndarray
            Binary mask (uint8, H×W) — 255 for damage, 0 for road.
        """
        method = method or self.cfg.method
        if method == "graphcut":
            return self._graphcut_segment(gray)
        else:
            return self._threshold_segment(gray)

    # ── Method A: Adaptive Thresholding ─────────────────────────────────

    def _threshold_segment(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding + morphological opening/closing.

        Steps:
        1. Adaptive Gaussian threshold (binary inverse) — dark regions
           become foreground (candidate damage).
        2. Morphological opening — erode then dilate to remove small noise.
        3. Morphological closing — dilate then erode to fill gaps in
           damage regions.
        4. Filter out contours smaller than min_contour_area.
        """
        cfg = self.cfg

        # Step 1 — Adaptive threshold (binary inverse: dark → white)
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=cfg.adaptive_block,
            C=cfg.adaptive_c,
        )

        # Step 2 & 3 — Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Step 4 — Discard tiny contours (noise)
        mask = self._filter_small_contours(closed, cfg.min_contour_area)
        return mask

    # ── Method B: Graph-Cut / MRF ───────────────────────────────────────

    def _graphcut_segment(self, gray: np.ndarray) -> np.ndarray:
        """MRF-based binary segmentation via scipy max-flow.

        Energy formulation:
            E(L) = Σ_i D_i(l_i)  +  λ · Σ_{i,j∈N} V_{ij}(l_i, l_j)

        Data term D_i:
            Class statistics are estimated using the adaptive-threshold
            method's output as a seed — pixels flagged as damage by the
            fast threshold pass define the "pothole" class, and the rest
            define "road".  This avoids the pitfall of raw Otsu intensity
            splitting, which labels the entire dark road surface as damage.
            For each pixel we compute the Gaussian NLL under each class
            model, plus a class-prior penalty:
                D_i(l) = NLL_l(I_i) − log(π_l)

        Smoothness term V_{ij}:
            Contrast-sensitive Ising model:
                V_{ij} = exp(-β · (I_i − I_j)²)
            Penalises label changes across regions of similar intensity,
            while allowing cuts at strong edges.

        NOTE: scipy.sparse.csgraph.maximum_flow requires integer edge
        capacities.  We multiply float costs by `graphcut_capacity_scale`
        (default 1000) and cast to int.  This preserves relative cost
        ordering with negligible quantisation error.
        """
        cfg = self.cfg
        orig_h, orig_w = gray.shape[:2]

        # ── Downscale for tractability ──────────────────────────────────
        small = self._resize_for_graphcut(gray)
        h, w = small.shape[:2]
        n_pixels = h * w
        logger.debug("Graph-Cut grid: %d×%d = %d nodes", h, w, n_pixels)

        small_f = small.astype(np.float64)

        # ── Seed from adaptive threshold ────────────────────────────────
        # Use the fast threshold method on the downscaled image to get an
        # initial damage estimate.  This gives us class statistics that
        # reflect actual local-contrast damage regions rather than a
        # naive global intensity split.
        seed_mask = self._threshold_segment(small)
        seed_damage = seed_mask == 255
        seed_road = ~seed_damage

        # Guard against empty classes
        if seed_damage.sum() == 0 or seed_road.sum() == 0:
            logger.warning("Seed mask has single class — returning threshold result")
            # Upscale the threshold result to original size
            return cv2.resize(
                self._threshold_segment(gray), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

        # ── Estimate class Gaussian models from seed ────────────────────
        pothole_pixels = small_f[seed_damage]
        road_pixels = small_f[seed_road]

        mu_p, sigma_p = pothole_pixels.mean(), max(pothole_pixels.std(), 1.0)
        mu_r, sigma_r = road_pixels.mean(), max(road_pixels.std(), 1.0)

        # Class priors — fraction of pixels in each class from seed.
        # Heavily biased toward road (damage is typically small),
        # which prevents the MRF from flooding the image with pothole labels.
        prior_p = max(seed_damage.sum() / n_pixels, 0.01)
        prior_r = max(seed_road.sum() / n_pixels, 0.01)

        # ── Data term (negative log-likelihood + prior + bias) ──────────
        # Source edge = cost of calling pixel "road"  (high → pushes to pothole)
        # Sink edge   = cost of calling pixel "pothole" (high → pushes to road)
        flat = small_f.ravel()
        nll_road = 0.5 * ((flat - mu_r) / sigma_r) ** 2 + np.log(sigma_r)
        nll_pothole = 0.5 * ((flat - mu_p) / sigma_p) ** 2 + np.log(sigma_p)

        # Clamp NLL to prevent extreme values from dominating the min-cut.
        max_nll = 10.0
        nll_road = np.clip(nll_road, 0, max_nll)
        nll_pothole = np.clip(nll_pothole, 0, max_nll)

        # Add prior-based bias: subtract log(prior) makes the minority class
        # (pothole) harder to assign — a pixel must be much more pothole-like
        # than road-like to overcome this penalty.
        nll_road = nll_road - np.log(prior_r)
        nll_pothole = nll_pothole - np.log(prior_p)

        # ── Smoothness term (contrast-sensitive Ising) ──────────────────
        # β = 1 / (2 * <(I_i - I_j)²>)  averaged over all neighbor pairs
        # V_{ij} = exp(-β * (I_i - I_j)²)
        diff_right = (small_f[:, :-1] - small_f[:, 1:]) ** 2
        diff_down = (small_f[:-1, :] - small_f[1:, :]) ** 2
        mean_sq_diff = (diff_right.sum() + diff_down.sum()) / (
            diff_right.size + diff_down.size
        )
        beta = 1.0 / (2.0 * mean_sq_diff + 1e-6) * cfg.graphcut_beta_multiplier

        # ── Build capacity graph (vectorised) ───────────────────────────
        # Node layout:  0 = source,  1 = sink,  2..n_pixels+1 = pixel nodes
        SOURCE = 0
        SINK = 1
        OFFSET = 2
        n_nodes = n_pixels + 2

        scale = cfg.graphcut_capacity_scale
        lam = cfg.graphcut_lambda

        # --- Source / Sink edges (data term) ---
        # Standard s-t graph cut convention (Boykov & Kolmogorov):
        #   Source→pixel  = D_i(pothole)  — heavy → pixel stays on source (road) side
        #   Pixel→sink    = D_i(road)     — heavy → pixel stays on sink (pothole) side
        pixel_indices = np.arange(n_pixels, dtype=np.int32) + OFFSET
        src_caps = np.maximum((nll_pothole * scale).astype(np.int32), 0)  # cost of pothole
        snk_caps = np.maximum((nll_road * scale).astype(np.int32), 0)    # cost of road

        src_rows = np.full(n_pixels, SOURCE, dtype=np.int32)
        snk_cols = np.full(n_pixels, SINK, dtype=np.int32)

        # --- Neighbor edges (smoothness term, vectorised) ---
        # Right neighbors
        y_r, x_r = np.mgrid[0:h, 0:w - 1]
        idx_r = (y_r * w + x_r).ravel()
        nb_idx_r = (y_r * w + x_r + 1).ravel()
        diff_sq_r = (flat[idx_r] - flat[nb_idx_r]) ** 2
        v_cost_r = np.maximum((lam * np.exp(-beta * diff_sq_r) * scale).astype(np.int32), 0)

        # Down neighbors
        y_d, x_d = np.mgrid[0:h - 1, 0:w]
        idx_d = (y_d * w + x_d).ravel()
        nb_idx_d = ((y_d + 1) * w + x_d).ravel()
        diff_sq_d = (flat[idx_d] - flat[nb_idx_d]) ** 2
        v_cost_d = np.maximum((lam * np.exp(-beta * diff_sq_d) * scale).astype(np.int32), 0)

        # Assemble all edges
        all_rows = np.concatenate([
            src_rows, pixel_indices,
            idx_r + OFFSET, nb_idx_r + OFFSET,
            idx_d + OFFSET, nb_idx_d + OFFSET,
        ])
        all_cols = np.concatenate([
            pixel_indices, snk_cols,
            nb_idx_r + OFFSET, idx_r + OFFSET,
            nb_idx_d + OFFSET, idx_d + OFFSET,
        ])
        all_caps = np.concatenate([
            src_caps, snk_caps,
            v_cost_r, v_cost_r,
            v_cost_d, v_cost_d,
        ])

        # Filter out zero-capacity edges
        nonzero = all_caps > 0
        all_rows = all_rows[nonzero]
        all_cols = all_cols[nonzero]
        all_caps = all_caps[nonzero]

        # Build sparse matrix and solve max-flow (= min-cut)
        graph = csr_matrix(
            (all_caps, (all_rows, all_cols)),
            shape=(n_nodes, n_nodes),
        )

        result = maximum_flow(graph, SOURCE, SINK)

        # ── Extract labels from residual graph ──────────────────────────
        # Nodes reachable from source in residual = "road" (label 0)
        # Unreachable nodes = "pothole" (label 1)
        residual = result.flow
        residual_cap = graph - residual

        reachable = self._bfs_reachable(residual_cap, SOURCE, n_nodes)

        labels = np.zeros(n_pixels, dtype=np.uint8)
        for i in range(n_pixels):
            if (i + OFFSET) not in reachable:
                labels[i] = 255

        mask_small = labels.reshape(h, w)

        # ── Upscale mask to original resolution ─────────────────────────
        mask_full = cv2.resize(
            mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        # ── Morphological cleanup ───────────────────────────────────────
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_full = self._filter_small_contours(mask_full, cfg.min_contour_area)

        return mask_full

    # ── Helpers ─────────────────────────────────────────────────────────

    def _resize_for_graphcut(self, gray: np.ndarray) -> np.ndarray:
        """Downscale image for Graph-Cut processing."""
        h, w = gray.shape[:2]
        max_dim = self.cfg.graphcut_resolution
        longest = max(h, w)
        if longest <= max_dim:
            return gray
        scale = max_dim / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _bfs_reachable(residual_cap: csr_matrix, source: int,
                       n_nodes: int) -> set:
        """BFS on the residual-capacity graph to find source-reachable nodes."""
        visited = set()
        queue = [source]
        visited.add(source)
        while queue:
            node = queue.pop(0)
            # Iterate over neighbours with positive residual capacity
            row_start = residual_cap.indptr[node]
            row_end = residual_cap.indptr[node + 1]
            for idx in range(row_start, row_end):
                nb = residual_cap.indices[idx]
                cap = residual_cap.data[idx]
                if cap > 0 and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited

    @staticmethod
    def _filter_small_contours(mask: np.ndarray,
                               min_area: int) -> np.ndarray:
        """Zero out contour regions smaller than *min_area* pixels."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        clean = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(clean, [cnt], -1, 255, thickness=cv2.FILLED)
        return clean
