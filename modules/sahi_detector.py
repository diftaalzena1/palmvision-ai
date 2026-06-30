# modules/sahi_detector.py
import streamlit as st
import torch
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from config import MODEL_PATH, BOX_RAW
from modules.image_utils import compute_optimal_slice


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------
def _get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model Loaders
# NOTE: Cache is keyed on model_path + device ONLY (NOT confidence).
#       Confidence is set at inference time, not load time.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sahi_model() -> AutoDetectionModel:
    """Load SAHI/AutoDetectionModel once per session. Confidence applied at predict-time."""
    device = _get_device()
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=MODEL_PATH,
        confidence_threshold=0.01,   # permissive — filtered per-call via conf_thres
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_yolo_model() -> YOLO:
    """Load bare YOLO model once per session."""
    return YOLO(MODEL_PATH)


def get_model(use_sahi: bool, conf_thres: float):
    """Return the appropriate model (conf_thres only used for routing, not caching)."""
    if use_sahi:
        return load_sahi_model()
    return load_yolo_model()


# ---------------------------------------------------------------------------
# Lightweight Slice Preview Generator
# ---------------------------------------------------------------------------
def generate_slice_previews(
    img_np: np.ndarray,
    slice_size: int,
    overlap_ratio: float,
    max_previews: int = 12,
    border_color: tuple = (0, 200, 100),
    border_thickness: int = 2,
) -> tuple[list, int]:
    """
    Manually crop the input image into slice previews using the same grid logic
    as SAHI (overlap approximation).  Returns (list_of_crops, total_slice_count).

    - Uses numpy slicing only — no heavy computation.
    - Safe for Streamlit rerun (pure function, no side-effects).
    - Returns at most `max_previews` images for UI display.
    """
    h, w = img_np.shape[:2]
    step = max(1, int(slice_size * (1.0 - overlap_ratio)))

    y_starts = list(range(0, max(1, h - slice_size + 1), step))
    x_starts = list(range(0, max(1, w - slice_size + 1), step))

    # Ensure last slice covers the edge
    if not y_starts or y_starts[-1] + slice_size < h:
        y_starts.append(max(0, h - slice_size))
    if not x_starts or x_starts[-1] + slice_size < w:
        x_starts.append(max(0, w - slice_size))

    # Deduplicate and sort
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    total_slices = len(y_starts) * len(x_starts)

    # Build preview list (up to max_previews, evenly sampled)
    all_coords = [(y, x) for y in y_starts for x in x_starts]
    if len(all_coords) > max_previews:
        indices = np.linspace(0, len(all_coords) - 1, max_previews, dtype=int)
        sampled = [all_coords[i] for i in indices]
    else:
        sampled = all_coords

    previews = []
    for (y0, x0) in sampled:
        y1 = min(y0 + slice_size, h)
        x1 = min(x0 + slice_size, w)
        crop = img_np[y0:y1, x0:x1].copy()
        if border_thickness > 0:
            cv2.rectangle(
                crop,
                (0, 0),
                (crop.shape[1] - 1, crop.shape[0] - 1),
                border_color,
                border_thickness,
            )
        previews.append(crop)

    return previews, total_slices


# ---------------------------------------------------------------------------
# Core Detection Pipeline
# ---------------------------------------------------------------------------
def process_image(img_np: np.ndarray, params: dict) -> dict:
    """
    Run full detection pipeline and return a UI-compatible result dict.

    Output keys:
        annotated       : np.ndarray — image with drawn boxes + centroids
        boxes           : list[list[int]] — [x1,y1,x2,y2] raw detections
        coords          : list[list[int]] — [cx,cy] final tree centroids
        rec_coords      : list[list[int]] — [cx,cy] recommended planting spots
        slice_previews  : list[np.ndarray] — real image crops (max 12)
        slice_counts    : list[int] — per-slice detection count (parallel to slice_previews)
        slice_boxes     : list[list[list[int]]] — per-slice local bounding boxes [x1,y1,x2,y2]
        img_shape       : tuple[int,int] — (h, w)
        total_trees     : int
        raw_detections  : int
        avg_conf        : float
    """
    h, w = img_np.shape[:2]

    # ── Pre-processing ────────────────────────────────────────────────────
    if params.get("use_clahe", False):
        from modules.image_utils import apply_lab_clahe
        img_proc = apply_lab_clahe(img_np)
    else:
        img_proc = img_np.copy()

    annotated = img_proc.copy()
    boxes_all: list[list[int]] = []
    confs_all: list[float] = []
    slice_previews: list[np.ndarray] = []
    slice_counts: list[int] = []          # per-preview detection count
    slice_boxes: list[list[list[int]]] = []  # per-preview local boxes

    # ── Inference ─────────────────────────────────────────────────────────
    if params.get("use_sahi", False):
        max_slices   = params.get("max_slices", 25)
        min_slice    = params.get("min_slice", 256)
        overlap_ratio = params.get("overlap_ratio", 0.2)
        conf_thres   = params.get("conf_thres", 0.25)

        slice_size = compute_optimal_slice(h, w, max_slices=max_slices, min_slice=min_slice)

        # Update confidence threshold on the cached model before inference
        model = params["model"]
        model.confidence_threshold = conf_thres

        result = get_sliced_prediction(
            img_proc,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )

        for obj in result.object_prediction_list:
            x1 = int(obj.bbox.minx)
            y1 = int(obj.bbox.miny)
            x2 = int(obj.bbox.maxx)
            y2 = int(obj.bbox.maxy)
            conf = float(obj.score.value)
            boxes_all.append([x1, y1, x2, y2])
            confs_all.append(conf)

        # ── Generate real slice previews (manual numpy cropping) ──────────
        previews, total_slices = generate_slice_previews(
            img_proc,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            max_previews=12,
        )
        slice_previews = previews

        # ── Compute per-slice local bounding boxes (for UI drawing) ───────
        # Recompute grid coordinates (identical to generate_slice_previews logic)
        step = max(1, int(slice_size * (1.0 - overlap_ratio)))
        y_starts = sorted(set(list(range(0, max(1, h - slice_size + 1), step)) + [max(0, h - slice_size)]))
        x_starts = sorted(set(list(range(0, max(1, w - slice_size + 1), step)) + [max(0, w - slice_size)]))
        all_coords_grid = [(y, x) for y in y_starts for x in x_starts]

        # Sample exactly the same preview slices as generate_slice_previews
        if len(all_coords_grid) > 12:
            indices = np.linspace(0, len(all_coords_grid) - 1, 12, dtype=int)
            sampled_coords = [all_coords_grid[i] for i in indices]
        else:
            sampled_coords = all_coords_grid

        slice_boxes = []
        slice_counts = []
        for (sy0, sx0) in sampled_coords:
            sy1 = min(sy0 + slice_size, h)
            sx1 = min(sx0 + slice_size, w)
            local_boxes = []
            count = 0
            for (x1, y1, x2, y2) in boxes_all:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if sx0 <= cx < sx1 and sy0 <= cy < sy1:
                    count += 1
                    # Offset to local coordinates
                    lx1 = x1 - sx0
                    ly1 = y1 - sy0
                    lx2 = x2 - sx0
                    ly2 = y2 - sy0
                    # Clip to slice bounds (safety)
                    lx1 = max(0, min(lx1, sx1 - sx0))
                    ly1 = max(0, min(ly1, sy1 - sy0))
                    lx2 = max(0, min(lx2, sx1 - sx0))
                    ly2 = max(0, min(ly2, sy1 - sy0))
                    if lx2 > lx1 and ly2 > ly1:
                        local_boxes.append([lx1, ly1, lx2, ly2])
            slice_counts.append(count)
            slice_boxes.append(local_boxes)

        actual_total_slices = total_slices

    else:
        # ── YOLO Fallback ─────────────────────────────────────────────────
        model = params["model"]
        conf_thres = params.get("conf_thres", 0.25)
        results = model.predict(source=img_proc, conf=conf_thres, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes_all.append([int(x1), int(y1), int(x2), int(y2)])
            confs_all.append(float(box.conf[0]))
        actual_total_slices = 0
        slice_boxes = []  # no slices for YOLO mode

    # ── Draw bounding boxes — BEFORE DBSCAN (raw, all detections) ─────────
    # Color gradient KUNING (low conf) -> HIJAU (high conf), unchanged behavior.
    thickness = max(2, int(min(h, w) / 500))
    annotated_before = annotated.copy()
    for (x1, y1, x2, y2), conf in zip(boxes_all, confs_all):
        r = int(255 * (1 - conf))
        g = int(255 * (0.5 + 0.5 * conf))
        b = 0
        color = (r, g, b)
        cv2.rectangle(annotated_before, (x1, y1), (x2, y2), color, thickness)

    # ── DBSCAN Clustering ─────────────────────────────────────────────────
    dbscan_labels: list[int] = []
    dbscan_applied = bool(len(boxes_all) > 0 and params.get("use_dbscan", True))
    if dbscan_applied:
        from modules.dbscan_cluster import apply_dbscan
        total_trees, coords, dbscan_labels = apply_dbscan(
            boxes_all, params.get("eps_factor", 0.6), h, w
        )
    else:
        total_trees = len(boxes_all)
        coords = [[int((x1 + x2) / 2), int((y1 + y2) / 2)] for (x1, y1, x2, y2) in boxes_all]
        dbscan_labels = list(range(len(boxes_all)))  # each box is its own "cluster"

    merged_count = max(0, len(boxes_all) - total_trees)

    # ── Draw bounding boxes — AFTER DBSCAN (deduplicated result) ───────────
    # - Duplicate raw boxes that DBSCAN merged into ONE final tree are drawn
    #   thin/red (CENTROID_CLUSTER) so the user can see exactly what got
    #   collapsed.
    # - The single resulting tree per cluster is drawn as a solid green
    #   (FINAL_TREE) box, centered on the cluster centroid, sized to match
    #   the average member box.
    from config import CENTROID_CLUSTER, FINAL_TREE

    annotated_after = annotated.copy()
    if dbscan_labels:
        labels_arr = np.array(dbscan_labels)
        boxes_arr = np.array(boxes_all)

        # Map each cluster label -> list of member box indices
        unique_lbls = sorted(set(dbscan_labels))
        label_to_indices = {lbl: np.where(labels_arr == lbl)[0] for lbl in unique_lbls}

        coord_i = 0  # walks coords in the same order apply_dbscan produced them
        for lbl in [l for l in unique_lbls if l != -1] or unique_lbls:
            member_idx = label_to_indices.get(lbl, [])
            if len(member_idx) == 0:
                continue
            members = boxes_arr[member_idx]

            if len(member_idx) > 1:
                # Duplicates merged by DBSCAN — show the raw boxes thin/red
                for (x1, y1, x2, y2) in members:
                    cv2.rectangle(
                        annotated_after, (int(x1), int(y1)), (int(x2), int(y2)),
                        CENTROID_CLUSTER, max(1, thickness - 1),
                    )

            # Final deduplicated tree box: centered on centroid, sized to
            # the average of its merged member boxes.
            avg_w = float((members[:, 2] - members[:, 0]).mean())
            avg_h = float((members[:, 3] - members[:, 1]).mean())
            cx = float(((members[:, 0] + members[:, 2]) / 2).mean())
            cy = float(((members[:, 1] + members[:, 3]) / 2).mean())
            fx1, fy1 = int(cx - avg_w / 2), int(cy - avg_h / 2)
            fx2, fy2 = int(cx + avg_w / 2), int(cy + avg_h / 2)
            cv2.rectangle(annotated_after, (fx1, fy1), (fx2, fy2), FINAL_TREE, thickness)

    # `annotated` kept for backward-compat: defaults to the AFTER view since
    # that is the model's final, trusted output.
    annotated = annotated_after

    # ── Recommendation (empty spots) ─────────────────────────────────────
    rec_coords: list[list[int]] = []
    if total_trees > 0 and coords:
        from modules.recommendation import find_empty_spots
        rec_coords = find_empty_spots(coords, (h, w))

    avg_conf = float(np.mean(confs_all)) if confs_all else 0.0

    return {
        "annotated":         annotated,          # = annotated_after (back-compat)
        "annotated_before":  annotated_before,    # raw detections, pre-DBSCAN
        "annotated_after":   annotated_after,     # deduplicated, post-DBSCAN
        "merged_count":      merged_count,        # duplicate boxes collapsed by DBSCAN
        "dbscan_applied":    dbscan_applied,       # False = DBSCAN was off for this run
        "boxes":          boxes_all,
        "coords":         coords,
        "rec_coords":     rec_coords,
        "slice_previews": slice_previews,
        "slice_counts":   slice_counts,
        "slice_boxes":    slice_boxes,          # NEW: per-slice local boxes
        "total_slices":   actual_total_slices,
        "img_shape":      (h, w),
        "total_trees":    total_trees,
        "raw_detections": len(boxes_all),
        "avg_conf":       avg_conf,
    }
