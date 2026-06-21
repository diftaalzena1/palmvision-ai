# modules/dbscan_cluster.py
import numpy as np
from sklearn.cluster import DBSCAN


def apply_dbscan(
    boxes_all: list[list[int]],
    eps_factor: float,
    img_h: int,
    img_w: int,
) -> tuple[int, list[list[int]]]:
    """
    Apply DBSCAN clustering on bounding box centroids.

    Args:
        boxes_all:  List of [x1, y1, x2, y2] bounding boxes.
        eps_factor: Scaling factor for adaptive eps (typical range 0.3–1.0).
        img_h:      Image height (used as fallback floor for eps).
        img_w:      Image width (used as fallback floor for eps).

    Returns:
        total_trees:  Number of unique tree clusters (noise excluded).
        coords:       List of [cx, cy] centroid coordinates per cluster.
    """
    if not boxes_all:
        return 0, []

    boxes_np = np.array(boxes_all, dtype=np.float32)

    # Centroids: shape (N, 2)
    centroids = np.stack([
        (boxes_np[:, 0] + boxes_np[:, 2]) / 2.0,
        (boxes_np[:, 1] + boxes_np[:, 3]) / 2.0,
    ], axis=1)

    # Adaptive eps: based on median bbox diagonal
    # Median is more robust than mean against outlier super-large boxes
    box_widths  = boxes_np[:, 2] - boxes_np[:, 0]
    box_heights = boxes_np[:, 3] - boxes_np[:, 1]
    avg_bbox_size = float(np.median(np.maximum(box_widths, box_heights)))

    # Floor prevents eps from collapsing to near-zero on tiny detections
    # Ceiling prevents over-merging on very large images with huge boxes
    eps_raw = eps_factor * avg_bbox_size
    eps_floor = max(10.0, float(min(img_h, img_w)) * 0.005)   # at least 0.5% of shorter side
    eps_ceil  = float(min(img_h, img_w)) * 0.15               # at most 15% of shorter side
    eps = float(np.clip(eps_raw, eps_floor, eps_ceil))

    # min_samples=1 ensures every point belongs to some cluster (no forced noise)
    # This is correct for detection deduplication: we trust the model, not neighbors
    db = DBSCAN(eps=eps, min_samples=1).fit(centroids)
    labels = db.labels_

    # Collect unique labels, explicitly EXCLUDING noise (-1)
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]

    if not unique_labels:
        # Fallback: all detections are noise — return each centroid as-is
        coords = [[int(cx), int(cy)] for (cx, cy) in centroids]
        return len(coords), coords

    # Compute per-cluster centroid as mean of member points
    coords: list[list[int]] = []
    for lbl in unique_labels:
        members = centroids[labels == lbl]
        cx = float(members[:, 0].mean())
        cy = float(members[:, 1].mean())
        coords.append([int(round(cx)), int(round(cy))])

    total_trees = len(coords)
    return total_trees, coords