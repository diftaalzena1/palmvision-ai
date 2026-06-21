# modules/image_utils.py
import cv2
import numpy as np
from PIL import Image

def apply_lab_clahe(img):
    """Apply LAB + CLAHE enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge((L, A, B))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def compute_optimal_slice(img_h, img_w, max_slices=25, min_slice=256, max_slice=1024):
    """Compute optimal slice size based on image dimensions."""
    slice_size = max_slice
    while True:
        n_h = int(np.ceil(img_h / slice_size))
        n_w = int(np.ceil(img_w / slice_size))
        if n_h * n_w <= max_slices or slice_size <= min_slice:
            break
        slice_size = int(slice_size * 0.8)
    return slice_size