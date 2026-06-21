# config.py
import streamlit as st
import os

# CONSTANTS
MODEL_PATH = "models/model_clahe_v4.pt"
BOX_RAW = (170, 170, 170)
CENTROID_CLUSTER = (255, 0, 0)
FINAL_TREE = (0, 255, 0)
MAX_BATCH_WARNING = 20
DEMO_FOLDER = "demo"

def get_recommendation(status):
    """Return recommendation based on status."""
    if status == "UNDERPOPULATED":
        return "Perlu penanaman ulang (replanting) di area kosong"
    elif status == "OVERPOPULATED":
        return "Perlu pemanenan atau penjarangan (thinning)"
    else:  # OPTIMAL
        return "Lanjutkan pengelolaan normal"