# modules/ui_components.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from config import DEMO_FOLDER, MAX_BATCH_WARNING

def inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

        :root {
            --forest:    #081a09;
            --canopy:    #122d14;
            --fern:      #1a4a1e;
            --leaf:      #256b2e;
            --sage:      #358040;
            --mist:      #b8d8bb;
            --foam:      #e8f4ea;
            --canvas:    #f2f9f3;
            --gold:      #b8861e;
            --amber:     #cc9a28;
            --gold-lt:   #f0d080;
            --ivory:     #fdfaf2;
            --chalk:     #ffffff;
            --ink:       #0a1c0b;
            --ash:       #4e6a52;
            --ash-lt:    #7a9e7e;
            --orange:    #d97706;
            --orange-lt: #fef3c7;
            --r-xs:  6px;
            --r-sm:  10px;
            --r-md:  16px;
            --r-lg:  24px;
            --r-xl:  32px;
            --shadow-xs: 0 1px 3px rgba(8,26,9,0.06);
            --shadow-sm: 0 2px 8px rgba(8,26,9,0.08);
            --shadow-md: 0 6px 24px rgba(8,26,9,0.12);
            --shadow-lg: 0 16px 48px rgba(8,26,9,0.18);
            --t: all 0.26s cubic-bezier(0.4,0,0.2,1);
        }

        /* ── BASE ─────────────────────────────────────────────────── */
        html, body, .stApp {
            font-family: 'DM Sans', sans-serif !important;
            background: var(--canvas) !important;
            color: var(--ink) !important;
        }
        header[data-testid="stHeader"],
        [data-testid="stDecoration"] { display: none !important; }
        .block-container {
            padding-top: 0.75rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1400px !important;
        }

        /* ── FIX: Remove spurious white panels from Streamlit columns ── */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        /* Kill any auto-added white bg on column children */
        [data-testid="column"] > div > div {
            background: transparent !important;
        }
        /* st.image: remove padding/margin wrapper */
        [data-testid="stImage"] {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 0 !important;
        }
        [data-testid="stImage"] > img {
            display: block !important;
            border-radius: var(--r-md) !important;
        }
        /* Remove white background from plotly chart container */
        [data-testid="stPlotlyChart"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        [data-testid="stPlotlyChart"] > div {
            background: transparent !important;
        }
        /* Remove excess margin from element-container wrappers */
        .element-container {
            margin-bottom: 0 !important;
        }
        /* ===== SIDEBAR TOGGLE BUTTON (FIX VISIBILITY) ===== */

        /* container header biar tombolnya tidak “hilang” */
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0.15) !important;
            backdrop-filter: blur(10px);
        }

        /* tombol toggle sidebar */
        [data-testid="stBaseButton-headerNoPadding"],
        button[data-testid="baseButton-headerNoPadding"],
        button[kind="header"],
        [data-testid="stToolbar"] button {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(204,154,40,0.35) !important;
            border-radius: 10px !important;
            padding: 0.35rem !important;
            color: #f0d080 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 9999 !important;
        }

        /* hover state */
        [data-testid="stBaseButton-headerNoPadding"]:hover,
        button[kind="header"]:hover {
            background: rgba(255,255,255,0.15) !important;
            border-color: rgba(204,154,40,0.8) !important;
            transform: scale(1.05);
        }

        /* icon hamburger */
        svg {
            fill: #f0d080 !important;
        }
        /* ══════════════════════════════════════════════════════════
           SIDEBAR — BASE SHELL
        ══════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: linear-gradient(168deg, #050f06 0%, #091509 35%, #0c1e0d 65%, #0f2410 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.05) !important;
            box-shadow: 4px 0 32px rgba(0,0,0,0.30) !important;
        }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
        [data-testid="stSidebar"] [class*="st-emotion-cache"],
        [data-testid="stSidebar"] [class*="css-"],
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stNumberInput,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stToggle,
        [data-testid="stSidebar"] .stCheckbox {
            background-color: transparent !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] div {
            background-color: transparent !important;
        }

        /* ── LOGO ─────────────────────────────────────────────────── */
        .sidebar-logo {
            text-align: center;
            padding: 1.2rem 0 0.6rem;  /* ← tambah padding atas & bawah */
        }
        .sidebar-logo-icon {
            font-size: 2.5rem;         /* ← sedikit lebih besar */
            line-height: 1;
            margin-bottom: 0.3rem;     /* ← tambah jarak ke teks */
            filter: drop-shadow(0 0 14px rgba(184,134,30,0.5));
        }
        .sidebar-logo-name {
            font-weight: 700;
            font-size: 1.3rem;         /* ← sedikit lebih besar */
            color: #fdfaf2 !important;
            letter-spacing: -0.3px;
            margin-top: 0.2rem;        /* ← kurangi margin top */
        }
        .sidebar-logo-tag {
            font-size: 0.65rem;        /* ← sedikit lebih besar */
            font-weight: 600;
            color: var(--amber) !important;
            letter-spacing: 1.5px;     /* ← sedikit kurang rapat */
            text-transform: uppercase;
            margin-top: 0.2rem;        /* ← tambah jarak dari nama */
        }
        .sidebar-divider {
            margin: 0.75rem 0;
            border: none;
            border-top: 1px solid rgba(255,255,255,0.06) !important;
        }
        .sidebar-label {
            font-size: 0.58rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.6px !important;
            color: rgba(184,134,30,0.65) !important;
            margin: 0.5rem 0 0.4rem !important;
            padding-left: 0.1rem !important;
            display: block !important;
        }

        /* ══════════════════════════════════════════════════════════
           ABOUT CARDS
        ══════════════════════════════════════════════════════════ */
        .sb-card {
            background: linear-gradient(
                145deg,
                rgba(255,255,255,0.06),
                rgba(255,255,255,0.02)
            );
            border: 1px solid rgba(240,208,128,0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            backdrop-filter: blur(12px);
            box-shadow:
                0 6px 18px rgba(0,0,0,0.22),
                inset 0 1px 0 rgba(255,255,255,0.04);
            transition: var(--t);
        }
        .sb-card:hover {
            transform: translateY(-2px);
            border-color: rgba(240,208,128,0.35);
            box-shadow:
                0 10px 24px rgba(0,0,0,0.28),
                0 0 18px rgba(204,154,40,0.08);
        }
        .sb-card-title {
            font-size: 0.58rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #f0d080;
            margin-bottom: 0.6rem;
        }
        .sb-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.35rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .sb-key {
            font-size: 0.78rem;
            color: rgba(255,255,255,0.78) !important;
            font-weight: 500;
        }
        .sb-val {
            font-size: 0.82rem;
            font-weight: 700;
            color: #f0d080;
        }
        .sb-note {
            font-size: 0.67rem;
            color: rgba(255,255,255,0.82) !important;
            line-height: 1.60;
            margin-top: 0.35rem;
        }
        .sb-warn {
            font-size: 0.67rem;
            color: #f0d080 !important;
            line-height: 1.60;
            margin-top: 0.20rem;
            font-weight: 600;
        }

        /* ══════════════════════════════════════════════════════════
           RADIO NAVIGATION
        ══════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] div[data-testid="stRadio"] {
            background: transparent !important;
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div {
            background: transparent !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > label {
            display: none !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
            width: 100% !important;
            gap: 2px !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {
            display: flex !important;
            align-items: center !important;
            gap: 0.65rem !important;
            width: 100% !important;
            background: transparent !important;
            border-radius: 8px !important;
            margin: 2px 0 !important;
            padding: 0.60rem 0.85rem !important;
            transition: all 0.18s ease !important;
            border-left: 2px solid transparent !important;
            cursor: pointer !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            color: rgba(255,255,255,0.62) !important;
            letter-spacing: 0.1px !important;
            box-sizing: border-box !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] * {
            color: inherit !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            background-color: rgba(255,255,255,0.06) !important;
            color: rgba(255,255,255,0.92) !important;
            border-left: 2px solid rgba(204,154,40,0.55) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background: rgba(26,74,30,0.50) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-left: 2px solid var(--amber) !important;
            box-shadow: inset 0 1px 1px rgba(0,0,0,0.12) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
            display: none !important;
        }

        /* ══════════════════════════════════════════════════════════
           SIDEBAR FORM ELEMENTS
        ══════════════════════════════════════════════════════════ */
        /* FIX: Sidebar checkbox label text (Detection tab) */
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stCheckbox div,
        [data-testid="stSidebar"] .stCheckbox span {
            color: rgba(255,255,255,0.85) !important;
        }

        /* khusus label utama checkbox */
        [data-testid="stSidebar"] label p {
            color: rgba(255,255,255,0.85) !important;
        } 
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stToggle label {
            color: rgba(200,220,202,0.78) !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
        [data-testid="stSidebar"] [data-testid="stTextInput"] input {
            background-color: rgba(255,255,255,0.07) !important;
            border: 1px solid rgba(255,255,255,0.13) !important;
            color: #ffffff !important;
            border-radius: var(--r-sm) !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] [data-baseweb="select"] {
            background-color: rgba(255,255,255,0.07) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: var(--r-sm) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] p {
            color: #ffffff !important;
        }
        [data-baseweb="popover"] [data-baseweb="menu"],
        [data-baseweb="popover"] [role="listbox"] {
            background-color: #0f2410 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
            background-color: var(--amber) !important;
            border-color: var(--gold-lt) !important;
        }
        [data-testid="stSidebar"] .stToggle [role="switch"][aria-checked="true"] {
            background-color: var(--amber) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="checkbox"][aria-checked="true"] > div:first-child {
            background-color: var(--amber) !important;
            border-color: var(--amber) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:first-child {
            border: 1.5px solid rgba(255,255,255,0.45) !important;
            background-color: rgba(255,255,255,0.04) !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: rgba(255,255,255,0.88) !important;
            border-radius: 8px !important;
        }
        .reset-btn-wrap .stButton > button {
            background: rgba(204,154,40,0.06) !important;
            border: 1px solid rgba(204,154,40,0.30) !important;
            color: rgba(204,154,40,0.85) !important;
        }

        /* ══════════════════════════════════════════════════════════
           HERO BAND
        ══════════════════════════════════════════════════════════ */
        .hero-band {
            background: linear-gradient(112deg, #050f06 0%, #122d14 45%, #1a4a1e 100%);
            border-radius: var(--r-xl);
            padding: 2rem 2.5rem 1.8rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        .hero-band::before {
            content: '';
            position: absolute;
            right: -40px; top: -40px;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(184,134,30,0.18) 0%, transparent 65%);
            pointer-events: none;
        }
        .hero-band::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, var(--amber), transparent);
        }
        .hero-band h1 {
            color: var(--ivory) !important;
            font-size: 2rem !important;
            margin: 0 0 0.35rem !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 20px rgba(0,0,0,0.3);
        }
        .hero-band p {
            color: #7fad85 !important;
            font-size: 0.88rem;
            margin: 0;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(184,134,30,0.15);
            border: 1px solid rgba(184,134,30,0.45);
            color: var(--gold-lt) !important;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 1.2px;
            padding: 0.22rem 0.8rem;
            border-radius: 50px;
            margin-bottom: 0.9rem;
        }

        /* ══════════════════════════════════════════════════════════
           IMAGE CARD (Upload grid)
        ══════════════════════════════════════════════════════════ */
        .image-card {
            background: var(--chalk);
            border-radius: var(--r-lg);
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
            transition: var(--t);
            border: 1px solid rgba(204,154,40,0.2);
        }
        .image-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-md);
            border-color: rgba(204,154,40,0.5);
        }
        .metadata-grid {
            margin-top: 0.8rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--foam);
        }
        .image-card .stCheckbox { margin-bottom: 0.5rem; }
        .image-card .stImage    { margin: 0.5rem 0; }

        /* ══════════════════════════════════════════════════════════
           SECTION HEADERS
        ══════════════════════════════════════════════════════════ */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin: 2rem 0 1.2rem;
        }
        .section-badge {
            width: 34px; height: 34px;
            background: linear-gradient(135deg, var(--fern), var(--forest));
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.75rem; color: white;
            flex-shrink: 0;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--canopy);
            letter-spacing: -0.2px;
        }
        .section-line {
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, #c0d8c2, transparent);
        }

        /* ── BUTTONS ──────────────────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, var(--fern) 0%, var(--forest) 100%) !important;
            color: var(--ivory) !important;
            border-radius: 50px !important;
            padding: 0.5rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            border: none !important;
            box-shadow: 0 3px 10px rgba(8,26,9,0.22) !important;
            transition: var(--t) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(8,26,9,0.28) !important;
        }
        [data-testid="stDownloadButton"] > button {
            background: transparent !important;
            border: 1.5px solid var(--leaf) !important;
            color: var(--fern) !important;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background: var(--leaf) !important;
            color: white !important;
        }

        /* ── RESULT PANEL ─────────────────────────────────────────── */
        .result-panel {
            background: linear-gradient(112deg, #060f07 0%, var(--canopy) 55%, var(--fern) 100%);
            border-radius: var(--r-xl);
            padding: 1.6rem 2rem;
            margin: 1rem 0;
            box-shadow: var(--shadow-lg);
        }
        .result-label {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.9px;
            color: #6b9a70;
            margin-bottom: 0.25rem;
        }
        .result-value {
            font-family: 'DM Sans', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            color: var(--ivory);
            line-height: 1;
            letter-spacing: -1px;
        }
        .result-unit {
            font-size: 0.75rem;
            color: var(--gold-lt);
            margin-top: 0.25rem;
            font-weight: 600;
        }

        /* ══════════════════════════════════════════════════════════
           DETAIL ANALYTICS — METADATA RIBBON
        ══════════════════════════════════════════════════════════ */
        .detail-meta-ribbon {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0;
            background: linear-gradient(90deg, #f0f8f1, #e8f4ea);
            border: 1px solid #c4ddc6;
            border-radius: var(--r-md);
            padding: 0.7rem 1.2rem;
            margin-bottom: 1.2rem;
        }
        .dmr-item {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
            padding: 0 1.1rem;
        }
        .dmr-item:first-child { padding-left: 0; }
        .dmr-item:last-child  { padding-right: 0; }
        .dmr-label {
            font-size: 0.57rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--ash);
        }
        .dmr-value {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--fern);
        }
        .dmr-sep {
            width: 1px;
            height: 32px;
            background: #b8d8bb;
            flex-shrink: 0;
        }

        /* ══════════════════════════════════════════════════════════
           DETAIL ANALYTICS — METRIC CHIPS ROW
        ══════════════════════════════════════════════════════════ */
        .metric-chip-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.2rem;
        }
        .metric-chip {
            border-radius: var(--r-md);
            padding: 0.85rem 1rem;
            text-align: center;
            border: 1px solid transparent;
            transition: var(--t);
        }
        .metric-chip:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
        .chip-primary {
            background: linear-gradient(135deg, var(--fern), var(--canopy));
            border-color: var(--fern);
        }
        .chip-primary .chip-val  { color: var(--ivory); }
        .chip-primary .chip-lbl  { color: rgba(255,255,255,0.65); }
        .chip-secondary {
            background: var(--chalk);
            border-color: #c4ddc6;
        }
        .chip-secondary .chip-val { color: var(--fern); }
        .chip-secondary .chip-lbl { color: var(--ash); }
        .chip-accent {
            background: linear-gradient(135deg, #fef9ee, #fdf3d0);
            border-color: rgba(204,154,40,0.4);
        }
        .chip-accent .chip-val { color: var(--gold); }
        .chip-accent .chip-lbl { color: var(--ash); }
        .chip-val {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            line-height: 1;
        }
        .chip-lbl {
            font-size: 0.62rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 0.25rem;
        }

        /* ── Detection image wrapper: no extra whitespace ────────── */
        .det-image-wrap {
            border-radius: var(--r-md);
            overflow: hidden;
            margin-bottom: 1rem;
            line-height: 0;
            background: transparent;
        }
        .det-image-wrap img {
            display: block !important;
            width: 100% !important;
            border-radius: var(--r-md) !important;
        }

        /* ══════════════════════════════════════════════════════════
           TABS, EXPANDER, METRICS, DATAFRAME
        ══════════════════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: var(--foam);
            padding: 0.3rem;
            border-radius: 50px;
            border: 1px solid #c4ddc6;
            width: fit-content;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 50px;
            padding: 0.35rem 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--ash) !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--chalk) !important;
            color: var(--fern) !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid #cde0ce !important;
            border-radius: var(--r-md) !important;
            margin-bottom: 0.75rem !important;
        }
        [data-testid="stExpander"] summary {
            background: linear-gradient(90deg, var(--foam), #f0f8f1) !important;
            padding: 0.8rem 1rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetric"] {
            background: var(--chalk);
            border: 1px solid #d0e4d2;
            border-radius: var(--r-md);
            padding: 0.8rem !important;
        }
        .stDataFrame {
            border-radius: var(--r-md);
            border: 1px solid #c4ddc6;
        }

        /* ══════════════════════════════════════════════════════════
           INFO STRIP & SLICE BADGE
        ══════════════════════════════════════════════════════════ */
        .info-strip {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: linear-gradient(90deg, #e8f4ea, #f2f9f3);
            border-left: 3px solid var(--leaf);
            border-radius: var(--r-sm);
            padding: 0.65rem 1rem;
            font-size: 0.82rem;
            color: var(--fern);
            margin: 0.6rem 0;
        }
        .slice-badge {
            display: inline-flex;
            gap: 1rem;
            background: var(--foam);
            padding: 0.3rem 1.1rem;
            border-radius: 50px;
            margin-bottom: 1rem;
            border: 1px solid #c0dcc3;
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* ══════════════════════════════════════════════════════════
           EXPORT PRODUCT CARDS (Refactored)
        ══════════════════════════════════════════════════════════ */
        .export-product-card {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            background: var(--chalk);
            border: 1px solid rgba(204,154,40,0.3);
            border-radius: var(--r-lg);
            padding: 1rem 1.2rem 0.8rem;
            margin-bottom: 0.6rem;
            transition: var(--t);
            box-shadow: var(--shadow-xs);
        }
        .export-product-card:hover {
            border-color: var(--amber);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
        /* Full-card variant for Section 04 */
        .export-full-card {
            padding: 1.2rem 1.4rem 1rem;
        }
        .epc-icon-wrap {
            width: 44px; height: 44px;
            border-radius: var(--r-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.7rem;
            letter-spacing: 0.5px;
            color: white;
            flex-shrink: 0;
        }
        .epc-zip { background: linear-gradient(135deg, #256b2e, #122d14); }
        .epc-xls { background: linear-gradient(135deg, #217346, #155c2f); }
        .epc-img { background: linear-gradient(135deg, #1a4a1e, #0c2b10); }
        .epc-csv { background: linear-gradient(135deg, #b8861e, #8a6010); }
        .epc-body {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            flex: 1;
            min-width: 0;
        }
        .epc-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--fern);
            white-space: nowrap;
        }
        .epc-desc {
            font-size: 0.72rem;
            color: var(--ash);
            line-height: 1.4;
        }
        .epc-meta {
            font-size: 0.65rem;
            color: var(--ash-lt);
            margin-top: 0.2rem;
            font-weight: 500;
        }

        /* ══════════════════════════════════════════════════════════
           EMPTY STATE & FOOTER
        ══════════════════════════════════════════════════════════ */
        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            background: var(--chalk);
            border-radius: var(--r-xl);
            margin-top: 2rem;
            border: 2px dashed #c0dcc3;
        }
        .empty-icon  { font-size: 3.5rem; margin-bottom: 0.8rem; }
        .empty-title { font-weight: 700; color: var(--leaf); font-size: 1.1rem; }
        .footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--ash);
            font-size: 0.7rem;
            margin-top: 2.5rem;
            padding: 1rem 0 0.5rem;
            border-top: 1px solid #c8e0ca;
        }
        .footer-brand { font-weight: 700; color: var(--fern) !important; }
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #c0dcc3, transparent);
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# RENDER FUNCTIONS
# -------------------------------------------------------------------
def render_section_header(number, title):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-badge">{number}</div>
            <div class="section-title">{title}</div>
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🌴</div>
            <div class="empty-title">No Images Uploaded</div>
            <p style="color:#4e6a52; font-size:0.85rem; max-width:500px; margin:0 auto;">
                Upload aerial oil palm plantation images, or click <strong>"Load Demo Image"</strong> in the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# INTERACTIVE MAP
# -------------------------------------------------------------------
def plot_interactive_map(coords, rec_coords, img_shape, max_size=700):
    if not coords:
        return None
    h, w = img_shape
    if w >= h:
        plot_width  = max_size
        plot_height = int(max_size * h / w)
    else:
        plot_height = max_size
        plot_width  = int(max_size * w / h)

    df = pd.DataFrame(coords, columns=["x", "y"])
    fig = px.scatter(
        df, x="x", y="y",
        title="Tree Distribution Map",
        labels={"x": "X (px)", "y": "Y (px)"},
        width=plot_width, height=plot_height,
    )
    fig.update_traces(
        marker=dict(size=8, color="#256b2e", line=dict(width=1.5, color="white"), opacity=0.85),
        name="Detected Trees",
    )
    if rec_coords:
        df_rec = pd.DataFrame(rec_coords, columns=["x", "y"])
        fig.add_trace(
            go.Scatter(
                x=df_rec["x"], y=df_rec["y"],
                mode="markers",
                name="Recommended Spots",
                marker=dict(size=8, color="#cc9a28", symbol="circle",
                            line=dict(width=1.5, color="white"), opacity=0.85),
            )
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        plot_bgcolor="#f2f9f3",
        paper_bgcolor="rgba(0,0,0,0)",   # transparent so host bg shows through
        font=dict(family="DM Sans, sans-serif", color="#122d14", size=11),
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=True, gridcolor="#ddeedd"),
        yaxis=dict(scaleanchor="x", showgrid=True, gridcolor="#ddeedd"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=10, r=10, t=44, b=10),
    )
    return fig

# -------------------------------------------------------------------
# DEMO HELPERS
# -------------------------------------------------------------------
def get_demo_images():
    if not os.path.exists(DEMO_FOLDER):
        return []
    files = [f for f in os.listdir(DEMO_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return sorted(files)

def load_demo_image_from_file(filename):
    path = os.path.join(DEMO_FOLDER, filename)
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None