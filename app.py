# app.py — Oil Palm Detection System (Premium UI v3 - Card Layout)
import streamlit as st
import time
import gc
import numpy as np
import cv2
from PIL import Image, ImageDraw
import pandas as pd
import io
from datetime import datetime

# =========================
# PAGE CONFIG (WAJIB PALING ATAS)
# =========================
st.set_page_config(
    page_title="PalmVision AI",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config
from modules.sahi_detector import get_model, process_image
from modules.exporter import export_to_zip, export_to_excel
from modules.ui_components import (
    plot_interactive_map,
    get_demo_images,
    load_demo_image_from_file,
    inject_global_css,
    render_section_header,
    render_empty_state,
)

inject_global_css()

# ===================================================================
# METADATA KEBUN: DROPDOWN OPTIONS
# ===================================================================
PT_OPTIONS = ["PT SSM", "PT SAE", "PT MAL"]

KEBUN_OPTIONS = {
    "PT SSM": ["Kebun KDE", "Kebun MJE"],
    "PT SAE": ["Kebun Pak Mayam"],
    "PT MAL": ["Kebun OKU Selatan"]
}

AFDELING_OPTIONS = {
    "Kebun KDE": ["AFD 1", "AFD 2", "AFD 3", "AFD 4", "AFD 5"],
    "Kebun MJE": ["AFD 1", "AFD 2", "AFD 3", "AFD 4"],
    "Kebun Pak Mayam": ["AFD 1", "AFD 2", "AFD 3", "AFD 4"],
    "Kebun OKU Selatan": ["AFD 1"]
}

DEFAULT_PT = "PT SAE"
DEFAULT_KEBUN = "Kebun Pak Mayam"
DEFAULT_AFDELING = "AFD 1"
DEFAULT_NAMA_BLOK = "A14"
DEFAULT_LUAS = 10.0

# -------------------------------------------------------------------
# DEFAULT SETTINGS
# -------------------------------------------------------------------
DEFAULTS = {
    "use_sahi":   True,
    "conf_thres": 0.25,
    "use_clahe":  True,
    "use_dbscan": True,
    "eps_factor": 0.60,
    "land_type":  "Mineral",
}

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "processing_cancel" not in st.session_state:
    st.session_state.processing_cancel = False
if "settings" not in st.session_state:
    st.session_state.settings = DEFAULTS.copy()
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False

# -------------------------------------------------------------------
# CACHED MODEL LOADER
# FIX: Cache is keyed on `use_sahi` only — NOT on `conf_thres`.
#      Confidence is applied at inference time inside process_image.
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model(use_sahi: bool):
    """Load model once per session. Confidence threshold is NOT a cache key."""
    return get_model(use_sahi, conf_thres=None)


# -------------------------------------------------------------------
# HELPER: Draw bounding boxes for potential planting spots (TASK 2)
# -------------------------------------------------------------------
def draw_potential_boxes(annotated_array, rec_coords, boxes):
    """
    Overlay orange bounding boxes (match distribution map color) on potential planting spots.
    """
    if not boxes or not rec_coords:
        return annotated_array
    
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    sizes = [(w + h) / 2 for w, h in zip(widths, heights)]
    avg_size = float(np.mean(sizes))
    
    target_size = int(0.7 * avg_size)
    target_size = max(15, min(100, target_size))
    half = target_size // 2
    
    img = annotated_array.copy()
    
    # Gunakan RGB (karena gambar mungkin dalam format RGB)
    ORANGE_RGB = (255, 165, 0)      # Oranye terang
    DARK_ORANGE_RGB = (200, 80, 0)  # Outline oranye gelap
    
    for (cx, cy) in rec_coords:
        x0 = int(cx - half)
        y0 = int(cy - half)
        x1 = int(cx + half)
        y1 = int(cy + half)
        
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img.shape[1] - 1, x1)
        y1 = min(img.shape[0] - 1, y1)
        
        # Outline hitam tebal untuk kontras
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=3)
        # Kotak oranye (RGB)
        cv2.rectangle(img, (x0, y0), (x1, y1), ORANGE_RGB, thickness=2)
    
    return img


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🌴</div>
            <div class="sidebar-logo-name">PalmVision AI</div>
            <div class="sidebar-logo-tag">Agronomic Decision Support</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    selected = st.radio(
        label="Navigation",
        options=["About", "Detection", "Guide"],
        index=0,
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if selected == "About":
        st.markdown(
            """
            <div class="sb-card">
            <div class="sb-card-title">Model Performance</div>
            <div class="sb-row"><span class="sb-key">Precision</span><span class="sb-val">96.3%</span></div>
            <div class="sb-row"><span class="sb-key">Recall</span><span class="sb-val">97.4%</span></div>
            <div class="sb-row"><span class="sb-key">mAP@50</span><span class="sb-val">98.6%</span></div>
            <div class="sb-row"><span class="sb-key">mAP@50-95</span><span class="sb-val">58.6%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sb-card">
            <div class="sb-card-title">Dataset</div>
            <div class="sb-row"><span class="sb-key">Train</span><span class="sb-val">839 img</span></div>
            <div class="sb-row"><span class="sb-key">Valid</span><span class="sb-val">240 img</span></div>
            <div class="sb-row"><span class="sb-key">Test</span><span class="sb-val">120 img</span></div>
            <div class="sb-note">PT SAE Saraswanti + Roboflow (TBM kecil)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sb-card">
              <div class="sb-card-title">Stack</div>
              <div class="sb-row"><span class="sb-key">Model</span><span class="sb-val">YOLOv8</span></div>
              <div class="sb-row"><span class="sb-key">Slicing</span><span class="sb-val">SAHI</span></div>
              <div class="sb-row"><span class="sb-key">Deduplication</span><span class="sb-val">DBSCAN</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sb-card">
            <div class="sb-card-title">Scope & Limitations</div>
            <div class="sb-note">
                Drone nadir, ketinggian ~100 m<br>
                Tanpa jalan / objek non-sawit<br>
                Kondisi mirip data latih
            </div>
            <div class="sb-warn">
                Hasil bersifat decision-support &mdash; tetap verifikasi lapangan.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif selected == "Detection":
        st.markdown('<span class="sidebar-label">Model Parameters</span>', unsafe_allow_html=True)
        st.session_state.settings["use_sahi"] = st.checkbox(
            "SAHI Sliced Inference",
            value=st.session_state.settings["use_sahi"],
            key="chk_sahi",
        )
        if st.session_state.settings["use_sahi"]:
            st.session_state.settings["conf_thres"] = st.slider(
                "Confidence Threshold",
                min_value=0.10, max_value=0.90,
                value=st.session_state.settings["conf_thres"],
                step=0.05,
                key="sl_conf",
            )
        st.markdown('<span class="sidebar-label">Processing</span>', unsafe_allow_html=True)
        st.session_state.settings["use_clahe"] = st.checkbox(
            "LAB + CLAHE Enhancement",
            value=st.session_state.settings["use_clahe"],
            key="chk_clahe",
        )
        st.session_state.settings["use_dbscan"] = st.checkbox(
            "Smart Counting (DBSCAN)",
            value=st.session_state.settings["use_dbscan"],
            key="chk_dbscan",
        )
        if st.session_state.settings["use_dbscan"]:
            st.session_state.settings["eps_factor"] = st.slider(
                "Clustering Sensitivity",
                min_value=0.30, max_value=1.00,
                value=st.session_state.settings["eps_factor"],
                step=0.05,
                key="sl_eps",
            )
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div class="reset-btn-wrap">', unsafe_allow_html=True)
        if st.button("Reset to Defaults", use_container_width=True, key="btn_reset_detection"):
            st.session_state.settings = DEFAULTS.copy()
            for k in ["chk_sahi", "chk_clahe", "chk_dbscan"]:
                if k in st.session_state:
                    del st.session_state[k]
            for k in ["sl_conf", "sl_eps"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:  # GUIDE
        st.markdown('<span class="sidebar-label">Demo Images</span>', unsafe_allow_html=True)
        demo_files = get_demo_images()
        if demo_files:
            selected_demo = st.selectbox("Pilih sampel:", demo_files, key="demo_select")
            if st.button("Load Demo Image", use_container_width=True, key="btn_load_demo"):
                with st.spinner(f"Memuat {selected_demo}..."):
                    img = load_demo_image_from_file(selected_demo)
                    if img:
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format="JPEG")
                        img_data = img_bytes.getvalue()
                        class DemoFile:
                            def __init__(self, name, data):
                                self.name = name
                                self._data = data
                            def read(self): return self._data
                            def getvalue(self): return self._data
                            @property
                            def size(self): return len(self._data)
                        st.session_state.demo_image = DemoFile(selected_demo, img_data)
                        st.session_state.demo_loaded = True
                        st.rerun()
        else:
            st.info("Tidak ada demo. Tambahkan gambar ke folder 'demo'.")

# -------------------------------------------------------------------
# HERO BAND
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-band">
        <div class="hero-pill">Agronomic Decision Support</div>
        <h1>PalmVision AI</h1>
        <p>Deteksi pohon sawit berbasis AI, analisis kerapatan tanam, dan rekomendasi titik tanam optimal</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# SECTION 01 — UPLOAD
# -------------------------------------------------------------------
render_section_header("01", "Upload Foto Udara")

uploaded_files = st.file_uploader(
    "Drag & drop atau browse foto udara (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Mendukung hingga 100 MB per file. Untuk gambar besar, aktifkan SAHI di tab Detection.",
    label_visibility="collapsed",
)

if st.session_state.get("demo_loaded", False) and "demo_image" in st.session_state:
    demo_file = st.session_state.demo_image
    if not uploaded_files:
        uploaded_files = [demo_file]
    else:
        if not any(f.name == demo_file.name for f in uploaded_files):
            uploaded_files = [demo_file] + list(uploaded_files)
    st.success(f"Demo '{demo_file.name}' berhasil dimuat — siap untuk deteksi.")

# ===================================================================
# TAMPILKAN CARD GAMBAR + METADATA PER GAMBAR (Premium Card Layout)
# ===================================================================
if uploaded_files:
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        if st.button("Select All", use_container_width=True):
            for i in range(len(uploaded_files)):
                st.session_state[f"img_sel_{i}"] = True
            st.rerun()
    with col_sel2:
        if st.button("Clear All", use_container_width=True):
            for i in range(len(uploaded_files)):
                st.session_state[f"img_sel_{i}"] = False
            st.rerun()

    selected_indices = []
    cols_per_row = 2
    num_files = len(uploaded_files)

    for i in range(0, num_files, cols_per_row):
        cols = st.columns(cols_per_row, gap="large")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < num_files:
                file = uploaded_files[idx]
                img = Image.open(file).convert("RGB")
                img_thumb = img.copy()
                img_thumb.thumbnail((300, 300))

                with col:
                    is_selected = st.checkbox(
                        f"**{file.name}**",
                        value=st.session_state.get(f"img_sel_{idx}", False),
                        key=f"img_sel_{idx}"
                    )
                    if is_selected:
                        selected_indices.append(idx)

                    st.image(img_thumb, use_container_width=True)

                    st.markdown('<div class="metadata-grid">', unsafe_allow_html=True)

                    # ROW 1
                    row1_col1, row1_col2 = st.columns(2)

                    with row1_col1:
                        pt_key = f"pt_{idx}"
                        current_pt = st.session_state.get(pt_key, DEFAULT_PT)
                        pt_val = st.selectbox(
                            "PT",
                            options=PT_OPTIONS,
                            index=PT_OPTIONS.index(current_pt) if current_pt in PT_OPTIONS else 0,
                            key=pt_key
                        )

                    with row1_col2:
                        kebun_options = KEBUN_OPTIONS.get(pt_val, [])
                        kebun_key = f"kebun_{idx}"
                        current_kebun = st.session_state.get(kebun_key, DEFAULT_KEBUN)
                        if current_kebun not in kebun_options:
                            current_kebun = kebun_options[0] if kebun_options else ""
                        kebun_val = st.selectbox(
                            "Kebun",
                            options=kebun_options,
                            index=kebun_options.index(current_kebun) if current_kebun in kebun_options else 0,
                            key=kebun_key
                        )

                    # ROW 2
                    row2_col1, row2_col2 = st.columns(2)

                    with row2_col1:
                        afdeling_options = AFDELING_OPTIONS.get(kebun_val, [])
                        afdeling_key = f"afdeling_{idx}"
                        current_afdeling = st.session_state.get(afdeling_key, DEFAULT_AFDELING)
                        if current_afdeling not in afdeling_options:
                            current_afdeling = afdeling_options[0] if afdeling_options else ""
                        afdeling_val = st.selectbox(
                            "Afdeling",
                            options=afdeling_options,
                            index=afdeling_options.index(current_afdeling) if current_afdeling in afdeling_options else 0,
                            key=afdeling_key
                        )

                    with row2_col2:
                        blok_val = st.text_input(
                            "Nama Blok",
                            value=st.session_state.get(f"blok_{idx}", DEFAULT_NAMA_BLOK),
                            key=f"blok_{idx}"
                        )

                    # ROW 3 - Luas
                    luas_val = st.number_input(
                        "Luas Blok (ha)",
                        min_value=0.1, max_value=500.0,
                        value=float(st.session_state.get(f"luas_{idx}", DEFAULT_LUAS)),
                        step=0.5,
                        key=f"luas_{idx}"
                    )

                    st.markdown('</div>', unsafe_allow_html=True)

    # ===================================================================
    # SECTION 02 — DETECT BUTTON
    # ===================================================================
    render_section_header("02", "Run Detection")

    if selected_indices:
        if st.button(
            f"🔍 Detect {len(selected_indices)} Image(s)",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.processing_cancel = False

            # FIX: Model cached on use_sahi only — confidence is passed as param
            model = load_cached_model(st.session_state.settings["use_sahi"])

            params = {
                "use_sahi":   st.session_state.settings["use_sahi"],
                "conf_thres": st.session_state.settings["conf_thres"],
                "use_clahe":  st.session_state.settings["use_clahe"],
                "use_dbscan": st.session_state.settings["use_dbscan"],
                "eps_factor": st.session_state.settings["eps_factor"],
                "model":      model,
            }

            results = []
            status_container = st.status("Processing images...", expanded=True)
            progress_bar = st.progress(0)
            cancel_btn = st.button("Cancel", key="cancel_process")
            start_time = time.time()

            for i, idx in enumerate(selected_indices):
                if cancel_btn or st.session_state.processing_cancel:
                    st.session_state.processing_cancel = True
                    status_container.update(label="Cancelled by user", state="error")
                    break

                file = uploaded_files[idx]
                status_container.write(f"Processing {file.name} ({i+1}/{len(selected_indices)})")
                elapsed = time.time() - start_time
                if i > 0:
                    est_remaining = (elapsed / i) * (len(selected_indices) - i)
                    status_container.write(f"Estimated remaining: {est_remaining:.1f} sec")

                pt_val       = st.session_state.get(f"pt_{idx}", DEFAULT_PT)
                kebun_val    = st.session_state.get(f"kebun_{idx}", DEFAULT_KEBUN)
                afdeling_val = st.session_state.get(f"afdeling_{idx}", DEFAULT_AFDELING)
                blok_val     = st.session_state.get(f"blok_{idx}", DEFAULT_NAMA_BLOK)
                luas_val     = st.session_state.get(f"luas_{idx}", DEFAULT_LUAS)

                img = np.array(Image.open(file).convert("RGB"))
                res = process_image(img, params)
                res["filename"]     = file.name
                density             = res["total_trees"] / luas_val if luas_val > 0 else 0
                res["density"]      = density
                res["pt"]           = pt_val
                res["kebun"]        = kebun_val
                res["afdeling"]     = afdeling_val
                res["nama_blok"]    = blok_val
                res["luas_blok_ha"] = luas_val

                results.append(res)
                progress_bar.progress((i + 1) / len(selected_indices))
                gc.collect()

            if not st.session_state.processing_cancel:
                status_container.update(label="Detection complete!", state="complete")
                st.session_state.batch_results = results
                st.toast(f"{len(results)} images processed successfully!")
                st.rerun()

    # ===================================================================
    # DISPLAY RESULTS
    # ===================================================================
    if st.session_state.batch_results:
        render_section_header("02", "Analysis Summary")
        results_list = st.session_state.batch_results

        total_trees_all = sum(r["total_trees"] for r in results_list)
        avg_density     = sum(r["density"] for r in results_list) / len(results_list)
        total_rec       = sum(len(r.get("rec_coords", [])) for r in results_list)

        st.markdown(
            f"""
            <div class="result-panel">
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:2rem; text-align:center;">
                    <div>
                        <div class="result-label">Total Trees</div>
                        <div class="result-value">{total_trees_all:,}</div>
                        <div class="result-unit">detected across all images</div>
                    </div>
                    <div>
                        <div class="result-label">Average Density</div>
                        <div class="result-value">{avg_density:.1f}</div>
                        <div class="result-unit">trees / hectare</div>
                    </div>
                    <div>
                        <div class="result-label">Planting Spots</div>
                        <div class="result-value">{total_rec:,}</div>
                        <div class="result-unit">recommended locations</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Batch Summary Table
        st.markdown(
            """
            <div class="section-header" style="margin-top: 1.5rem;">
                <div class="section-badge">📋</div>
                <div class="section-title">Batch Summary</div>
                <div class="section-line"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        df_res = pd.DataFrame([{
            "Gambar":          r["filename"],
            "PT":              r["pt"],
            "Kebun":           r["kebun"],
            "Afdeling":        r["afdeling"],
            "Nama Blok":       r["nama_blok"],
            "Luas (ha)":       r["luas_blok_ha"],
            "Pohon":           r["total_trees"],
            "Kepadatan (/ha)": f"{r['density']:.1f}",
        } for r in results_list])
        styled_df = df_res.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#e8f4ea"), ("color", "#1a4a1e"),
                ("text-align", "center"), ("font-weight", "700"),
                ("border-bottom", "2px solid #b8d8bb"),
            ]}
        ])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ===================================================================
        # SECTION 03 — DETAILED ANALYTICS
        # ===================================================================
        render_section_header("03", "Detailed Analytics per Image")

        for idx, res in enumerate(results_list):
            with st.expander(
                f"{res['filename']}  ·  {res['total_trees']:,} trees  |  {res['kebun']} - {res['nama_blok']}",
                expanded=(idx == 0)
            ):
                # ── Metadata ribbon ──────────────────────────────────────
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="detail-meta-ribbon">
                        <div class="dmr-item"><span class="dmr-label">PT</span><span class="dmr-value">{res['pt']}</span></div>
                        <div class="dmr-sep"></div>
                        <div class="dmr-item"><span class="dmr-label">Kebun</span><span class="dmr-value">{res['kebun']}</span></div>
                        <div class="dmr-sep"></div>
                        <div class="dmr-item"><span class="dmr-label">Afdeling</span><span class="dmr-value">{res['afdeling']}</span></div>
                        <div class="dmr-sep"></div>
                        <div class="dmr-item"><span class="dmr-label">Blok</span><span class="dmr-value">{res['nama_blok']}</span></div>
                        <div class="dmr-sep"></div>
                        <div class="dmr-item"><span class="dmr-label">Luas</span><span class="dmr-value">{res['luas_blok_ha']:.2f} ha</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Metric chips row ──────────────────────────────────────
                st.markdown(
                    f"""
                    <div class="metric-chip-row">
                        <!-- Hijau: Final Trees (hasil akhir) -->
                        <div class="metric-chip chip-primary">
                            <div class="chip-val">{res['total_trees']:,}</div>
                            <div class="chip-lbl">Final Trees</div>
                        </div>
                        <!-- Merah: Duplicates Removed by DBSCAN -->
                        <div class="metric-chip chip-secondary">
                            <div class="chip-val">{res.get('merged_count', 0):,}{' (off)' if not res.get('dbscan_applied', True) else ''}</div>
                            <div class="chip-lbl">Duplicates Removed (DBSCAN)</div>
                        </div>
                        <!-- Kuning: Density (metrik penting) -->
                        <div class="metric-chip chip-accent">
                            <div class="chip-val">{res['density']:.1f}</div>
                            <div class="chip-lbl">Trees / ha</div>
                        </div>
                        <!-- Kuning: Potential Spots (rekomendasi aksi) -->
                        <div class="metric-chip chip-accent">
                            <div class="chip-val">{len(res.get('rec_coords', []))}</div>
                            <div class="chip-lbl">Potential Spots</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Before/After DBSCAN toggle ─────────────────────────────
                dbscan_applied = res.get("dbscan_applied", True)

                if dbscan_applied:
                    view_mode = st.radio(
                        "Tampilan deteksi",
                        options=["After DBSCAN (final)", "Before DBSCAN (raw)", "Compare side-by-side"],
                        index=0,
                        horizontal=True,
                        key=f"view_mode_{idx}",
                        label_visibility="collapsed",
                    )
                    if res.get("merged_count", 0) > 0:
                        st.markdown(
                            f'<div class="info-strip">DBSCAN merged '
                            f'<strong>{res["merged_count"]} duplicate detection(s)</strong> '
                            f'into single trees — see boxes outlined in red on the "Before" view.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="info-strip">DBSCAN ran but found no duplicate detections to merge '
                            'for this image — raw and final counts are the same.</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    # DBSCAN was turned off in the sidebar for this run — don't
                    # pretend there's a before/after to compare, since both
                    # images are identical (raw detections only).
                    view_mode = "Before DBSCAN (raw)"
                    st.markdown(
                        '<div class="info-strip">⚠️ DBSCAN is <strong>disabled</strong> in sidebar settings — '
                        'showing raw detections only. Enable DBSCAN to deduplicate overlapping boxes.</div>',
                        unsafe_allow_html=True,
                    )

                # ── Image column (full width, clean) ──────────────────────
                show_potential = st.checkbox(
                    "Tampilkan potential planting spots",
                    key=f"show_potential_{idx}",
                )

                annotated_before_img = res.get("annotated_before", res["annotated"])
                annotated_after_img  = res.get("annotated_after", res["annotated"])

                st.markdown('<div class="det-image-wrap">', unsafe_allow_html=True)

                if view_mode == "Compare side-by-side":
                    cmp_col1, cmp_col2 = st.columns(2, gap="medium")
                    with cmp_col1:
                        st.image(annotated_before_img, use_container_width=True)
                        st.markdown(
                            "<div style='text-align:center;font-size:0.85rem;color:#4e6a52;'>"
                            f"Before DBSCAN · {res['raw_detections']:,} raw boxes</div>",
                            unsafe_allow_html=True,
                        )
                    with cmp_col2:
                        st.image(annotated_after_img, use_container_width=True)
                        st.markdown(
                            "<div style='text-align:center;font-size:0.85rem;color:#4e6a52;'>"
                            f"After DBSCAN · {res['total_trees']:,} final trees</div>",
                            unsafe_allow_html=True,
                        )
                elif show_potential and res.get("rec_coords"):
                    # Gunakan fungsi baru dengan bounding box adaptif (TASK 2)
                    base_img = annotated_after_img if view_mode.startswith("After") else annotated_before_img
                    overlay_arr = draw_potential_boxes(base_img, res["rec_coords"], res["boxes"])
                    st.image(
                        overlay_arr,
                        use_container_width=True,
                        caption="Detection + Potential Spots (kuning dengan outline hitam)",
                    )
                else:
                    base_img = annotated_after_img if view_mode.startswith("After") else annotated_before_img
                    st.image(
                        base_img,
                        use_container_width=True,
                    )

                    caption_text = (
                        "After DBSCAN — duplicates merged into final trees (red = removed, green = final)"
                        if view_mode.startswith("After")
                        else "Before DBSCAN — every raw detection shown, no deduplication"
                    )
                    st.markdown(
                        f"""
                        <div style="
                            margin-top:0.4rem;
                            font-size:0.85rem;
                            color:#4e6a52;
                            text-align:center;
                        ">
                            {caption_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # ── Tabs ──────────────────────────────────────────────────
                tab_map, tab_slices, tab_export = st.tabs(["Tree Map", "SAHI Slices", "Export"])

                with tab_map:
                    if res["coords"]:
                        fig = plot_interactive_map(res["coords"], res["rec_coords"], res["img_shape"])
                        st.plotly_chart(fig, use_container_width=True)
                        if res["rec_coords"]:
                            st.markdown(
                                f'<div class="info-strip">Found <strong>{len(res["rec_coords"])} potential spots</strong> for planting in empty areas.</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No coordinate data available for this image.")

                # ===========================================================
                # TAB SAHI SLICES (TASK 1)
                # ===========================================================
                with tab_slices:
                    slice_previews = res.get("slice_previews", [])
                    slice_counts   = res.get("slice_counts", [])
                    slice_boxes    = res.get("slice_boxes", [])
                    total_slices   = res.get("total_slices", 0)

                    if slice_previews and len(slice_previews) > 0:
                        total_det_display = sum(slice_counts) if slice_counts else res.get("raw_detections", 0)

                        st.markdown(
                            f'<div class="slice-badge">'
                            f'<span>Total Slices: {total_slices}</span>'
                            f'<span>{total_det_display} detections</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        displayed = slice_previews[:12]
                        slice_cols = st.columns(3)
                        for j, slice_img in enumerate(displayed):
                            det_count = slice_counts[j] if j < len(slice_counts) else 0
                            if j < len(slice_boxes) and slice_boxes[j]:
                                img_with_boxes = slice_img.copy()
                                for (lx1, ly1, lx2, ly2) in slice_boxes[j]:
                                    cv2.rectangle(img_with_boxes, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                                display_img = img_with_boxes
                            else:
                                display_img = slice_img
                            with slice_cols[j % 3]:
                                st.image(
                                    display_img,
                                    caption=f"Slice {j+1} · {det_count} det.",
                                    use_container_width=True,
                                )

                        if total_slices > 12:
                            st.caption(f"Showing 12 of {total_slices} total slices.")
                    else:
                        st.info("SAHI not used or no slices available for this image.")

                with tab_export:
                    img_bytes = io.BytesIO()
                    Image.fromarray(res["annotated"]).save(img_bytes, format="PNG")

                    ex_c1, ex_c2 = st.columns(2, gap="large")
                    with ex_c1:
                        st.markdown(
                            """
                            <div class="export-product-card">
                                <div class="epc-icon-wrap epc-img">PNG</div>
                                <div class="epc-body">
                                    <div class="epc-title">Annotated Image</div>
                                    <div class="epc-desc">Full-resolution PNG with detection bounding boxes</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.download_button(
                            "Download PNG", img_bytes.getvalue(),
                            f"{res['filename']}_annotated.png",
                            key=f"dl_img_{idx}", use_container_width=True,
                        )

                    with ex_c2:
                        if res["coords"]:
                            df_coords = pd.DataFrame(res["coords"], columns=["x", "y"])
                            df_coords["PT"]        = res["pt"]
                            df_coords["Kebun"]     = res["kebun"]
                            df_coords["Afdeling"]  = res["afdeling"]
                            df_coords["Nama_Blok"] = res["nama_blok"]
                            df_coords["Luas_ha"]   = res["luas_blok_ha"]
                            st.markdown(
                                """
                                <div class="export-product-card">
                                    <div class="epc-icon-wrap epc-csv">CSV</div>
                                    <div class="epc-body">
                                        <div class="epc-title">Coordinates CSV</div>
                                        <div class="epc-desc">Centroid coordinates + plantation metadata</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.download_button(
                                "Download CSV", df_coords.to_csv(index=False),
                                f"{res['filename']}_coords.csv",
                                key=f"dl_csv_{idx}", use_container_width=True,
                            )

        # ===================================================================
        # SECTION 04 — EXPORT ALL RESULTS
        # ===================================================================
        render_section_header("04", "Export All Results")

        ecol1, ecol2 = st.columns(2, gap="large")

        with ecol1:
            zip_data = export_to_zip(st.session_state.batch_results)
            st.markdown(
                f"""
                <div class="export-product-card export-full-card">
                    <div class="epc-icon-wrap epc-zip">ZIP</div>
                    <div class="epc-body">
                        <div class="epc-title">ZIP Archive</div>
                        <div class="epc-desc">All annotated images + coordinate CSVs in one package</div>
                        <div class="epc-meta">{len(st.session_state.batch_results)} file(s) · includes full metadata</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download ZIP", zip_data,
                f"batch_{datetime.now():%Y%m%d_%H%M%S}.zip",
                mime="application/zip", use_container_width=True,
            )

        with ecol2:
            excel_data = export_to_excel(st.session_state.batch_results)
            st.markdown(
                f"""
                <div class="export-product-card export-full-card">
                    <div class="epc-icon-wrap epc-xls">XLS</div>
                    <div class="epc-body">
                        <div class="epc-title">Excel Report</div>
                        <div class="epc-desc">Summary table + per-image coordinate sheets</div>
                        <div class="epc-meta">{len(st.session_state.batch_results)} sheet(s) · with plantation metadata</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download Excel", excel_data,
                f"report_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

else:
    render_empty_state()

st.markdown(
    """
    <div class="footer">
        <span class="footer-brand">PalmVision AI</span>
        <span>Agronomic Decision Support · 2026</span>
        <span>YOLOv8 + SAHI + DBSCAN</span>
    </div>
    """,
    unsafe_allow_html=True,
)
