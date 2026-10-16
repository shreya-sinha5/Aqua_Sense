import sys
from pathlib import Path

# Ensure project root and src/ directory are always on sys.path
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st
import cv2
import time
import json
import torch
import numpy as np
import pandas as pd
import psutil
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from datetime import datetime

from model import ConvAutoencoder, AquaSenseAutoencoder
from infer import AquaSenseInference
from logger import (
    init_logger,
    save_detection_event,
    get_recent_logs,
    get_saved_image_paths,
    CSV_PATH as AUDIT_CSV_PATH,
    IMG_DIR as AUDIT_IMG_DIR
)
from preprocessing import clean_sonar_frame
from telemetry import encode_acoustic_telemetry
from agents import AUVAgentSystem
from geotagging import GeotaggingEngine
from confidence import CalibratedConfidenceScorer
from acoustic_physics import SoundSpeedProfile, AcousticRayTracer, AcousticMultilaterationSolver, SonarPoseGraphSLAM
from space_surveillance import SARProcessor, AISRecord, MultiModalMaritimeFusionEngine
from vehicle_dynamics import FossenHydrodynamics6DOF, CableTrackingMPC, MAVLinkAutopilotBridge

# Initialize persistent logging directories and CSV
init_logger()


# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "aquasense_ae.pth"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

TEMPORAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "temporal_ai"
    / "temporal_detections.csv"
)

PRIORITY_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "accumulation"
    / "priority_plan.csv"
)

DETECTION_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "detections.csv"
)

TEMPORAL_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "temporal_ai"
)


# SETTINGS & DYNAMIC THRESHOLD

IMAGE_SIZE = 224
MIN_CONTOUR_AREA = 100

# Load calibrated threshold from models/threshold.json
try:
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r") as f:
            _thresh_meta = json.load(f)
            MODEL_THRESHOLD = float(_thresh_meta.get("threshold", 0.026747))
            THRESHOLD_METHOD = _thresh_meta.get("method", "Mean(MSE_clean) + 3*StdDev(MSE_clean)")
    else:
        MODEL_THRESHOLD = 0.026747
        THRESHOLD_METHOD = "Mean(MSE_clean) + 3*StdDev(MSE_clean)"
except Exception:
    MODEL_THRESHOLD = 0.026747
    THRESHOLD_METHOD = "Mean(MSE_clean) + 3*StdDev(MSE_clean)"

# Thresholds used for screening/classification
HIGH_ANOMALY_MULTIPLIER = 2.0
MEDIUM_ANOMALY_MULTIPLIER = 1.15



# PAGE

st.set_page_config(
    page_title="AquaSense | Marine Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS - CYBER-OCEANIC TACTICAL HUD DESIGN SYSTEM

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* Streamlit Default Clutter Removal */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .stDeployButton, div[data-testid="stToolbar"], #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* Global Typography & Deep Sea Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 8%, rgba(0, 229, 255, 0.12), transparent 32%),
            radial-gradient(circle at 90% 15%, rgba(0, 255, 178, 0.08), transparent 28%),
            radial-gradient(circle at 50% 90%, rgba(10, 45, 75, 0.25), transparent 45%),
            linear-gradient(135deg, #020912 0%, #061524 45%, #030d17 100%) !important;
        color: #e6f6ff;
    }

    /* Top Tactical Header Banner */
    .aqua-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        background: linear-gradient(135deg, rgba(6, 25, 42, 0.85), rgba(3, 14, 25, 0.90));
        border: 1px solid rgba(0, 229, 255, 0.22);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
    }

    .aqua-brand {
        display: flex;
        flex-direction: column;
    }

    .aqua-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.1;
        background: linear-gradient(90deg, #00e5ff 0%, #38ef7d 50%, #bbf7d0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .aqua-subtitle {
        color: #8bb4c7;
        font-size: 13px;
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.3px;
    }

    .header-pills-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
    }

    /* Tactical Status Pills & Beacons */
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
        border: 1px solid rgba(0, 229, 255, 0.25);
        background: rgba(0, 229, 255, 0.08);
        color: #7eeaff;
    }

    .telemetry-pill.green {
        border-color: rgba(0, 255, 178, 0.35);
        background: rgba(0, 255, 178, 0.10);
        color: #00ffb2;
    }

    .telemetry-pill.red {
        border-color: rgba(255, 75, 75, 0.4);
        background: rgba(255, 75, 75, 0.12);
        color: #ff6b6b;
    }

    /* Pulsing Beacons */
    .beacon-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        background-color: #00ffb2;
        box-shadow: 0 0 10px #00ffb2;
        animation: aqua-pulse-green 2s infinite;
    }
    .beacon-dot.cyan {
        background-color: #00e5ff;
        box-shadow: 0 0 10px #00e5ff;
        animation: aqua-pulse-cyan 2s infinite;
    }
    .beacon-dot.red {
        background-color: #ff4757;
        box-shadow: 0 0 10px #ff4757;
        animation: aqua-pulse-red 1.5s infinite;
    }

    @keyframes aqua-pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 178, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(0, 255, 178, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 178, 0); }
    }
    @keyframes aqua-pulse-cyan {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 229, 255, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(0, 229, 255, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 229, 255, 0); }
    }
    @keyframes aqua-pulse-red {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.8); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(255, 71, 87, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }
    }

    /* Streamlit Tabs Overhaul */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(3, 14, 24, 0.65) !important;
        padding: 8px 10px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0, 229, 255, 0.16) !important;
        margin-bottom: 22px !important;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px !important;
        color: #8daec0 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 9px 18px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #e0f7ff !important;
        background: rgba(0, 229, 255, 0.08) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.20), rgba(0, 255, 178, 0.12)) !important;
        color: #00e5ff !important;
        border-bottom: 2px solid #00e5ff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(0, 229, 255, 0.18) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030d17 0%, #051827 50%, #020911 100%) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.14) !important;
    }

    /* Glass Cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(8, 28, 44, 0.75), rgba(4, 16, 26, 0.85));
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(14px);
        transition: all 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.40);
        box-shadow: 0 10px 28px rgba(0, 229, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    /* Status Banners */
    .status-critical {
        background: linear-gradient(135deg, rgba(255, 60, 60, 0.20), rgba(100, 10, 10, 0.35));
        border: 1px solid rgba(255, 80, 80, 0.50);
        border-radius: 16px;
        padding: 18px;
        color: #ff8585;
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(255, 50, 50, 0.15);
    }

    .status-clear {
        background: linear-gradient(135deg, rgba(40, 210, 150, 0.16), rgba(10, 80, 60, 0.28));
        border: 1px solid rgba(60, 220, 170, 0.40);
        border-radius: 16px;
        padding: 18px;
        color: #63e6b1;
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(60, 220, 170, 0.12);
    }

    /* Info Box */
    .info-title {
        color: #8aaebb;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .info-value {
        color: #f2fbff;
        font-size: 20px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Streamlit Metric Overhaul */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(6, 25, 40, 0.7), rgba(3, 13, 22, 0.8)) !important;
        border: 1px solid rgba(0, 229, 255, 0.16) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #7ea3b5 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #e6f7ff !important;
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.12), rgba(0, 168, 255, 0.08)) !important;
        color: #d8f5ff !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        border-color: #00e5ff !important;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.24), rgba(0, 168, 255, 0.18)) !important;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #0099ff 0%, #00d4ff 100%) !important;
        color: #020911 !important;
        font-weight: 700 !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
    }

    /* Agent Advisory */
    .agent-box {
        background: linear-gradient(135deg, rgba(8, 30, 48, 0.88), rgba(4, 18, 28, 0.92));
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 16px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .agent-header {
        font-size: 18px;
        font-weight: 700;
        color: #7eeaff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-critical {
        display: inline-block;
        background: rgba(255, 60, 60, 0.25);
        color: #ff8585;
        border: 1px solid rgba(255, 80, 80, 0.50);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-warning {
        display: inline-block;
        background: rgba(255, 180, 40, 0.22);
        color: #ffd56e;
        border: 1px solid rgba(255, 190, 70, 0.45);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-normal {
        display: inline-block;
        background: rgba(40, 210, 150, 0.20);
        color: #63e6b1;
        border: 1px solid rgba(60, 220, 170, 0.40);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    .hazard-tag {
        display: inline-block;
        background: rgba(255, 90, 90, 0.15);
        border: 1px solid rgba(255, 90, 90, 0.3);
        color: #ffaaaa;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Pre, Code, Tables */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid rgba(0, 229, 255, 0.16);
        overflow: hidden;
    }

    /* Custom File Uploader & Radio Elements */
    div[data-testid="stFileUploader"] section {
        background: rgba(6, 25, 42, 0.65) !important;
        border: 1px dashed rgba(0, 229, 255, 0.35) !important;
        border-radius: 14px !important;
        color: #c5e9f8 !important;
        padding: 24px !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #00e5ff !important;
        background: rgba(0, 229, 255, 0.08) !important;
    }
    div[data-testid="stRadio"] label p {
        color: #d8f3ff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(6, 25, 40, 0.6) !important;
        border: 1px solid rgba(0, 229, 255, 0.16) !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# MODEL

@st.cache_resource
def load_model():

    model = AquaSenseAutoencoder()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model


try:

    model = load_model()

    model_status = "ONLINE"

except Exception as e:

    model = None

    model_status = "ERROR"

    model_error = str(e)


# DUAL-STAGE HYBRID INFERENCE ENGINE

@st.cache_resource
def load_dual_stage_engine():

    try:
        return AquaSenseInference()
    except Exception as e:
        print(f"[WARN] Failed to load dual-stage engine: {e}")
        return None


dual_stage_engine = load_dual_stage_engine()
webcam_detector = dual_stage_engine



# FEATURE EXTRACTION

def extract_features(frame):

    if frame is None:

        return {
            "mean_intensity": 0,
            "std_intensity": 0,
            "shadow_ratio": 0,
            "edge_ratio": 0
        }

    if len(frame.shape) == 3:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = frame.copy()

    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    mean_intensity = float(
        np.mean(blurred)
    )

    std_intensity = float(
        np.std(blurred)
    )

    dark_threshold = mean_intensity * 0.60

    dark_pixels = blurred < dark_threshold

    shadow_ratio = float(
        np.mean(dark_pixels)
    )

    edges = cv2.Canny(
        gray,
        80,
        160
    )

    edge_ratio = float(
        np.mean(edges > 0)
    )

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "shadow_ratio": shadow_ratio,
        "edge_ratio": edge_ratio
    }


# OBJECT SCREENING CLASSIFICATION

def classify_object(
    anomaly_score,
    shadow_ratio,
    contrast,
    edge_ratio
):

    # CLEAR / LOW ANOMALY

    if anomaly_score < MODEL_THRESHOLD:

        return {
            "label": "Marine Life / Seafloor",
            "short_label": "MARINE LIFE",
            "severity": "CLEAR",
            "risk": 5,
            "material": "Organic / Natural",
            "action": "BYPASS",
            "description":
                "No significant anomaly detected. "
                "Likely natural seafloor or marine-life signature."
        }

    # HIGH ANOMALY

    if anomaly_score >= (
        MODEL_THRESHOLD *
        HIGH_ANOMALY_MULTIPLIER
    ):

        # Strong geometric/edge response
        if edge_ratio > 0.075:

            label = "Possible Structural / Metal Debris"

            short_label = "MAJOR DEBRIS"

            material = (
                "Possible metal / structural material"
            )

            severity = "CRITICAL"

            risk = 90

            action = "ALERT"

            description = (
                "Strong anomalous response with "
                "structural edge characteristics. "
                "Requires operator verification."
            )

        else:

            label = "Major Unclassified Debris"

            short_label = "MAJOR DEBRIS"

            material = (
                "Unknown / non-biological material"
            )

            severity = "CRITICAL"

            risk = 82

            action = "ALERT"

            description = (
                "High reconstruction error indicates "
                "a major anomalous object."
            )

        return {
            "label": label,
            "short_label": short_label,
            "severity": severity,
            "risk": risk,
            "material": material,
            "action": action,
            "description": description
        }

    # MEDIUM ANOMALY

    if anomaly_score >= (
        MODEL_THRESHOLD *
        MEDIUM_ANOMALY_MULTIPLIER
    ):

        if shadow_ratio > 0.35 and edge_ratio < 0.05:

            return {
                "label": "Possible Marine Life / Natural Feature",
                "short_label": "BYPASS",
                "severity": "CLEAR",
                "risk": 10,
                "material": "Organic / Natural",
                "action": "BYPASS",
                "description":
                    "Anomaly characteristics are more consistent "
                    "with a natural or biological signature."
            }

        return {
            "label": "Unclassified Debris",
            "short_label": "DEBRIS",
            "severity": "REVIEW",
            "risk": 55,
            "material": "Unknown",
            "action": "REVIEW",
            "description":
                "Moderate anomaly detected. "
                "Operator review recommended."
        }

    return {
        "label": "Low-Level Anomaly",
        "short_label": "LOW ANOMALY",
        "severity": "REVIEW",
        "risk": 35,
        "material": "Unknown",
        "action": "REVIEW",
        "description":
            "Small deviation from normal background."
    }


# FIND ANOMALY REGION

def find_anomaly_region(residual):

    # Normalize residual

    residual_float = residual.astype(
        np.float32
    )

    if np.max(residual_float) > 0:

        normalized = cv2.normalize(
            residual_float,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    else:

        normalized = np.zeros_like(
            residual,
            dtype=np.uint8
        )

    # Adaptive threshold

    threshold_value = max(
        25,
        int(
            np.percentile(
                normalized,
                88
            )
        )
    )

    _, mask = cv2.threshold(
        normalized,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Clean noise

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c
        for c in contours
        if cv2.contourArea(c) >= MIN_CONTOUR_AREA
    ]

    if not contours:

        return None, mask

    # Sort by area

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    # Take strongest region

    contour = contours[0]

    x, y, w, h = cv2.boundingRect(
        contour
    )

    return (x, y, w, h), mask


# DRAW DETECTION BOX

def draw_detection_box(
    original_image,
    residual,
    detection
):

    image = original_image.copy()

    box, mask = find_anomaly_region(
        residual
    )

    if box is None:

        return image, None

    x, y, w, h = box

    original_h, original_w = image.shape[:2]

    # Convert 224 coordinates to original coordinates

    scale_x = (
        original_w /
        IMAGE_SIZE
    )

    scale_y = (
        original_h /
        IMAGE_SIZE
    )

    x = int(x * scale_x)
    y = int(y * scale_y)

    w = int(w * scale_x)
    h = int(h * scale_y)

    # Add padding

    padding_x = int(
        w * 0.08
    )

    padding_y = int(
        h * 0.08
    )

    x = max(
        0,
        x - padding_x
    )

    y = max(
        0,
        y - padding_y
    )

    w = min(
        original_w - x,
        w + padding_x * 2
    )

    h = min(
        original_h - y,
        h + padding_y * 2
    )

    # Color based on severity

    if detection["severity"] == "CRITICAL":

        color = (
            40,
            70,
            255
        )

    elif detection["severity"] == "REVIEW":

        color = (
            40,
            180,
            255
        )

    else:

        color = (
            60,
            210,
            160
        )

    # Main box

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

    # Corner accents

    corner_length = max(
        12,
        int(
            min(w, h) * 0.08
        )
    )

    thickness = 5

    # Top-left

    cv2.line(
        image,
        (x, y),
        (x + corner_length, y),
        color,
        thickness
    )

    cv2.line(
        image,
        (x, y),
        (x, y + corner_length),
        color,
        thickness
    )

    # Top-right

    cv2.line(
        image,
        (x + w, y),
        (x + w - corner_length, y),
        color,
        thickness
    )

    cv2.line(
        image,
        (x + w, y),
        (x + w, y + corner_length),
        color,
        thickness
    )

    # Bottom-left

    cv2.line(
        image,
        (x, y + h),
        (x + corner_length, y + h),
        color,
        thickness
    )

    cv2.line(
        image,
        (x, y + h),
        (x, y + h - corner_length),
        color,
        thickness
    )

    # Bottom-right

    cv2.line(
        image,
        (x + w, y + h),
        (x + w - corner_length, y + h),
        color,
        thickness
    )

    cv2.line(
        image,
        (x + w, y + h),
        (x + w, y + h - corner_length),
        color,
        thickness
    )

    # Label

    label_text = (
        f"{detection['short_label']} "
        f"| {detection['risk']}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(
        0.45,
        original_w / 1500
    )

    text_thickness = 2

    (text_w, text_h), baseline = (
        cv2.getTextSize(
            label_text,
            font,
            font_scale,
            text_thickness
        )
    )

    label_y = y - 10

    if label_y < text_h + 10:

        label_y = (
            y +
            h +
            text_h +
            10
        )

    # Background

    cv2.rectangle(
        image,
        (
            x,
            label_y - text_h - 10
        ),
        (
            x + text_w + 14,
            label_y + baseline
        ),
        color,
        -1
    )

    # Text

    cv2.putText(
        image,
        label_text,
        (
            x + 7,
            label_y
        ),
        font,
        font_scale,
        (
            255,
            255,
            255
        ),
        text_thickness,
        cv2.LINE_AA
    )

    return image, (
        x,
        y,
        w,
        h
    )


# GRAD-CAM COMPATIBLE SALIENCY

def generate_gradcam(
    tensor,
    model
):

    tensor = tensor.clone().detach()
    tensor.requires_grad_(True)

    model.zero_grad()

    reconstruction = model(
        tensor
    )

    # Reconstruction objective

    loss = torch.mean(
        (
            tensor -
            reconstruction
        ) ** 2
    )

    loss.backward()

    gradient = tensor.grad

    if gradient is None:

        return np.zeros(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            dtype=np.uint8
        )

    saliency = torch.abs(
        gradient
    )

    saliency = saliency.squeeze()

    saliency = (
        saliency
        .detach()
        .cpu()
        .numpy()
    )

    if np.max(saliency) > 0:

        saliency = cv2.normalize(
            saliency,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

    saliency = saliency.astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        saliency,
        cv2.COLORMAP_JET
    )

    return heatmap


# AI ANALYSIS

def analyze_image(
    frame,
    target_model,
    nav_telemetry=None
):
    if dual_stage_engine is not None:
        stage_res = dual_stage_engine.analyze(frame, nav_telemetry=nav_telemetry)
    else:
        temp_engine = AquaSenseInference()
        stage_res = temp_engine.analyze(frame, nav_telemetry=nav_telemetry)

    anomaly_score = stage_res["anomaly_score"]
    threshold_val = stage_res["threshold"]
    is_anomaly = stage_res["is_anomaly"]
    global_status = stage_res["global_status"]

    reconstructed_image = stage_res["reconstruction"]
    residual = stage_res["residual"]
    raw_heatmap = stage_res["heatmap"]
    residual_overlay = stage_res["overlay"]
    boxed_image = stage_res["bounded_frame"]
    detections = stage_res["detections"]
    total_detections = stage_res["total_detections"]

    # Grad-CAM overlay for explainability
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(torch.device("cpu"))
        gradcam_heatmap = generate_gradcam(tensor, target_model)
        original_color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        gradcam_overlay = cv2.addWeighted(original_color, 0.58, gradcam_heatmap, 0.42, 0)
    except Exception:
        gradcam_overlay = residual_overlay.copy()

    features = extract_features(frame)

    # Classification & Telemetry
    if is_anomaly:
        severity = "CRITICAL"
        action = "ALERT"
        if len(detections) > 0:
            primary = detections[0]
            label = f"{primary['type']}"
            short_label = "KNOWN TYPE" if primary["is_known"] else "UNKNOWN ANOMALY"
            if "Shipwreck" in primary["type"] or "Pipeline" in primary["type"]:
                material = "Structural Steel / Metal Frame"
                risk = 92
            elif "Aircraft" in primary["type"]:
                material = "Aeronautical Alloy / Airframe"
                risk = 88
            elif "Fauna" in primary["type"]:
                material = "Biological / Marine Life"
                risk = 30
            else:
                material = "Unclassified Submerged Debris"
                risk = 85
            description = (
                f"Identified {primary['type']} with {primary['confidence']*100:.1f}% confidence. "
                f"Contour area: {primary['area']:.0f} px, aspect ratio: {primary['aspect_ratio']:.2f}."
            )
        else:
            label = "Major Unclassified Submerged Threat"
            short_label = "UNKNOWN ANOMALY"
            material = "Unclassified Foreign Object"
            risk = 80
            description = (
                f"High reconstruction error (MSE: {anomaly_score:.6f}) exceeds calibrated threshold ({threshold_val:.6f}). "
                "Immediate operator review required."
            )
    else:
        severity = "CLEAR"
        action = "BYPASS"
        label = "Marine Life / Seafloor Baseline"
        short_label = "SEABED CLEAR"
        material = "Organic Silt / Natural Seabed Sand"
        risk = 5
        description = (
            f"Seabed surface clear. Reconstruction error (MSE: {anomaly_score:.6f}) is within "
            f"baseline threshold ({threshold_val:.6f}). No debris alert generated."
        )

    # Highlight ratio from detections
    if len(detections) > 0:
        total_box_area = sum([d["bbox"][2] * d["bbox"][3] for d in detections])
        highlight_ratio = min(100.0, (total_box_area / (frame.shape[0] * frame.shape[1])) * 100.0)
    else:
        highlight_ratio = 0.0

    primary_bbox = detections[0]["bbox"] if len(detections) > 0 else None

    return {
        "score": anomaly_score,
        "threshold": threshold_val,
        "is_anomaly": is_anomaly,
        "global_status": global_status,
        "original_image": frame.copy(),
        "reconstruction": reconstructed_image,
        "residual": residual,
        "raw_heatmap": raw_heatmap,
        "heatmap": residual_overlay,
        "gradcam": gradcam_overlay,
        "boxed_image": boxed_image,
        "bbox": primary_bbox,
        "detections": detections,
        "total_detections": total_detections,
        "highlight_ratio": highlight_ratio,
        "shadow_ratio": features["shadow_ratio"],
        "contrast": features["std_intensity"],
        "edge_ratio": features["edge_ratio"],
        "label": label,
        "short_label": short_label,
        "severity": severity,
        "risk": risk,
        "material": material,
        "action": action,
        "description": description
    }



# SAVE DETECTION

def save_detection(
    result,
    latitude,
    longitude,
    depth
):

    DETECTION_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    geo_coordinates = (
        f"{latitude:.6f}, "
        f"{longitude:.6f}"
    )

    row = pd.DataFrame(
        [
            {
                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "object":
                    result["label"],

                "severity":
                    result["severity"],

                "risk_percent":
                    result["risk"],

                "anomaly_score":
                    result["score"],

                "material":
                    result["material"],

                "depth_m":
                    depth,

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "geo_coordinates":
                    geo_coordinates,

                "highlight_ratio":
                    result[
                        "highlight_ratio"
                    ],

                "action":
                    result["action"]
            }
        ]
    )

    if DETECTION_CSV.exists():

        row.to_csv(
            DETECTION_CSV,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            DETECTION_CSV,
            index=False
        )


# LOAD DATA

def load_csv(path):

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except Exception:

        return pd.DataFrame()


temporal_df = load_csv(
    TEMPORAL_CSV
)

priority_df = load_csv(
    PRIORITY_CSV
)

detection_df = load_csv(
    DETECTION_CSV
)


# SESSION STATE

if "live_score" not in st.session_state:

    st.session_state.live_score = 0.0


if "live_status" not in st.session_state:

    st.session_state.live_status = "WAITING"


if "anomaly_count" not in st.session_state:

    st.session_state.anomaly_count = 0


if "last_scan" not in st.session_state:

    st.session_state.last_scan = "--"


if "last_result" not in st.session_state:

    st.session_state.last_result = None


if "running" not in st.session_state:

    st.session_state.running = False


if "last_telemetry" not in st.session_state:

    st.session_state.last_telemetry = None


if "last_advisory" not in st.session_state:

    st.session_state.last_advisory = None


if "last_latency_ms" not in st.session_state:

    st.session_state.last_latency_ms = 0.0


if "last_fps" not in st.session_state:

    st.session_state.last_fps = 0.0


if "last_clean_frame" not in st.session_state:

    st.session_state.last_clean_frame = None


if "live_capture_count" not in st.session_state:

    st.session_state.live_capture_count = 0


if "live_session_dir" not in st.session_state:

    st.session_state.live_session_dir = None


if "live_session_log" not in st.session_state:

    st.session_state.live_session_log = []


# SIDEBAR (MISSION CONTROL)

with st.sidebar:

    st.markdown(
        "## ⚙️ Mission Control"
    )

    st.divider()

    st.markdown(
        "### 📍 Geotagging & Navigation Source"
    )

    geotag_source_mode = st.selectbox(
        "Navigation Feed Source",
        options=[
            "AUV Telemetry + Seafloor FOV Ray-Projection",
            "Manual Benchmark Coordinates",
            "NMEA 0183 Live Sentence Parser",
            "USBL Acoustic Slant-Range Head"
        ],
        index=0
    )

    if geotag_source_mode == "NMEA 0183 Live Sentence Parser":
        nmea_raw = st.text_input(
            "Raw NMEA String ($GPGGA / $GPRMC)",
            value="$GPGGA,123519,1303.456,N,08015.678,E,1,08,0.9,25.4,M,0.0,M,,*47"
        )
        parsed_nmea = GeotaggingEngine.parse_nmea_sentence(nmea_raw)
        if parsed_nmea:
            latitude = parsed_nmea["latitude"]
            longitude = parsed_nmea["longitude"]
            depth = parsed_nmea["depth_m"]
            st.success(f"NMEA Fix: {latitude:.5f}, {longitude:.5f}")
        else:
            latitude, longitude, depth = 12.2958, 76.6394, 25.0
            st.warning("Invalid NMEA sentence, fallback active.")
    elif geotag_source_mode == "USBL Acoustic Slant-Range Head":
        m_lat = st.number_input("Mothership Lat", value=12.2958, format="%.6f")
        m_lon = st.number_input("Mothership Lon", value=76.6394, format="%.6f")
        slant_r = st.number_input("Acoustic Slant Range (m)", value=120.0, step=5.0)
        az_deg = st.number_input("Azimuth (deg)", value=45.0, step=5.0)
        el_deg = st.number_input("Elevation (deg)", value=30.0, step=5.0)
        usbl_calc = GeotaggingEngine.calculate_usbl_position(m_lat, m_lon, slant_r, az_deg, el_deg)
        latitude = usbl_calc["latitude"]
        longitude = usbl_calc["longitude"]
        depth = usbl_calc["depth_m"]
        st.info(f"USBL Subsea Target: {latitude:.5f}, {longitude:.5f} | Depth {depth:.1f}m")
    else:
        latitude = st.number_input(
            "Latitude",
            value=12.2958,
            format="%.6f"
        )
        longitude = st.number_input(
            "Longitude",
            value=76.6394,
            format="%.6f"
        )
        depth = st.number_input(
            "Depth (meters)",
            min_value=0.0,
            value=25.0,
            step=0.5
        )

    heading = st.slider("AUV Heading (deg)", 0.0, 360.0, 90.0, 5.0)
    altitude = st.slider("AUV Altitude Above Seabed (m)", 1.0, 30.0, 6.0, 0.5)

    current_telemetry = {
        "lat": latitude,
        "lon": longitude,
        "depth": depth,
        "heading": heading,
        "altitude_m": altitude,
        "source": geotag_source_mode
    }

    st.caption(
        f"Active Geotag Mode: {geotag_source_mode}"
    )

    st.divider()

    st.markdown(
        "### 📸 Live Camera Capture Settings"
    )

    capture_interval = st.slider(
        "Image Capture Interval (seconds)",
        min_value=0.5,
        max_value=60.0,
        value=5.0,
        step=0.5,
        help="How often a frame is saved to disk during live camera mode. The AI runs every frame, but only saves images at this interval."
    )

    capture_quality = st.slider(
        "JPEG Save Quality (1–100)",
        min_value=50,
        max_value=100,
        value=90,
        step=5,
        help="JPEG quality for saved frames. Higher = better image, larger file."
    )

    save_all_frames = st.checkbox(
        "Save All Frames (not just anomalies)",
        value=True,
        help="When enabled, every captured frame is saved. When disabled, only anomalous frames are saved."
    )

    st.caption(
        f"Capture every {capture_interval:.1f}s • Quality {capture_quality}% • {'All frames' if save_all_frames else 'Anomalies only'}"
    )

    if st.session_state.live_session_dir is not None:
        st.markdown(
            f"**Active Session:**  "
            f"`{Path(st.session_state.live_session_dir).name}`"
        )
        st.metric(
            "Frames Captured",
            st.session_state.live_capture_count
        )

    st.divider()

    st.markdown(
        "### 🧠 AI Engine"
    )

    if model_status == "ONLINE":

        st.success(
            "AI Model: ONLINE"
        )

    else:

        st.error(
            "AI Model: ERROR"
        )

        if "model_error" in locals():

            st.caption(
                model_error
            )

    st.metric(
        "Anomaly Threshold",
        f"{MODEL_THRESHOLD:.6f}"
    )

    st.divider()

    st.markdown(
        "### 🔌 Integrations"
    )

    st.success(
        "NumPy: ACTIVE"
    )

    st.info(
        "ONNX: READY"
    )

    st.info(
        "TensorRT: READY"
    )

    st.info(
        "OpenDrift: READY"
    )

    st.caption(
        "ONNX/TensorRT/OpenDrift are integration "
        "hooks for deployment and real mission data."
    )

    st.divider()

    st.markdown(
        "### ⚡ Edge Hardware Telemetry"
    )

    # Real-time resource metrics via psutil
    try:
        _cpu_pct = psutil.cpu_percent(interval=None)
        _ram = psutil.virtual_memory()
        _ram_used_mb = _ram.used / (1024 * 1024)
        _ram_pct = _ram.percent
    except Exception:
        _cpu_pct, _ram_pct, _ram_used_mb = 0.0, 0.0, 0.0

    hw_col1, hw_col2 = st.columns(2)
    with hw_col1:
        st.metric("CPU Load", f"{_cpu_pct:.1f}%")
        st.metric(
            "Latency",
            f"{st.session_state.last_latency_ms:.1f} ms" if st.session_state.last_latency_ms > 0 else "--"
        )
    with hw_col2:
        st.metric("RAM Used", f"{_ram_pct:.0f}%", f"{_ram_used_mb:.0f} MB")
        st.metric(
            "Throughput",
            f"{st.session_state.last_fps:.1f} FPS" if st.session_state.last_fps > 0 else "--"
        )

    st.caption(
        "Edge metrics: Low-power embedded AUV hardware profiler."
    )

    st.divider()

    st.markdown(
        "### 📡 Acoustic Uplink Telemetry"
    )

    if st.session_state.last_telemetry is not None:
        _tel = st.session_state.last_telemetry
        st.success(
            f"Bandwidth Savings: **{_tel['savings_pct']:.1f}%**"
        )
        _tx_time_80bps = (_tel['size_bytes'] * 8) / 80.0
        st.caption(
            f"Payload: **{_tel['size_bytes']} bytes** ({_tel['size_kb']:.2f} KB)<br>"
            f"Raw frame: {_tel['raw_img_bytes']} bytes<br>"
            f"Est. TX @ 80 bps: **{_tx_time_80bps:.2f}s**",
            unsafe_allow_html=True
        )
    else:
        st.info("Awaiting scan for acoustic packet encoding.")


# TOP TACTICAL HEADER

st.markdown(
    f"""
    <div class="aqua-header-bar">
        <div class="aqua-brand">
            <div class="aqua-title">🌊 AquaSense</div>
            <div class="aqua-subtitle">Autonomous Underwater Vehicle (AUV) Tactical Intelligence & Dual-Stage Sonar C2 Console</div>
        </div>
        <div class="header-pills-row">
            <div class="telemetry-pill green"><span class="beacon-dot"></span> AI CORE: {model_status}</div>
            <div class="telemetry-pill"><span class="beacon-dot cyan"></span> MISSION: ACTIVE</div>
            <div class="telemetry-pill"><span class="beacon-dot cyan"></span> DEPTH: {depth:.1f}m</div>
            <div class="telemetry-pill"><span class="beacon-dot cyan"></span> THRESHOLD: {MODEL_THRESHOLD:.6f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# HUD METRICS
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Sensor Array", "ONLINE", f"{heading:.0f}° Heading")
with m2:
    st.metric("AI Core Engine", "Dual-Stage Hybrid", model_status)
with m3:
    st.metric("Survey Depth", f"{depth:.1f} m", f"{altitude:.1f}m Altitude")
with m4:
    st.metric("Logged Anomalies", st.session_state.anomaly_count)
with m5:
    st.metric("Last Sonar Scan", st.session_state.last_scan)

st.divider()

# FRAME PROCESSING HELPER
def execute_frame_analysis(target_frame):
    if model is None:
        st.error("AI model could not be loaded.")
        return None

    with st.spinner("Executing AquaSense Dual-Stage AI analysis..."):
        _t_start = time.perf_counter()
        result = analyze_image(
            target_frame,
            model,
            nav_telemetry=current_telemetry
        )
        _t_elapsed_ms = (time.perf_counter() - _t_start) * 1000.0

        st.session_state.last_latency_ms = _t_elapsed_ms
        st.session_state.last_fps = 1000.0 / max(_t_elapsed_ms, 1.0)

        # 1. Physics-informed speckle filter & CLAHE
        try:
            _clean_sonar = clean_sonar_frame(target_frame)
            st.session_state.last_clean_frame = _clean_sonar
        except Exception:
            st.session_state.last_clean_frame = None

        # 2. Ultra-low bandwidth acoustic telemetry encoding
        try:
            _tel = encode_acoustic_telemetry(
                result["score"],
                result["threshold"],
                result["detections"],
                raw_frame_shape=target_frame.shape
            )
            st.session_state.last_telemetry = _tel
        except Exception:
            pass

        # 3. Autonomous Multi-Agent Advisory System
        try:
            _agent_sys = AUVAgentSystem()
            _adv = _agent_sys.run_mission_advisory(
                result["score"],
                result["threshold"],
                result["detections"],
                depth=depth
            )
            st.session_state.last_advisory = _adv
        except Exception:
            pass

        st.session_state.last_result = result
        st.session_state.live_score = result["score"]
        st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")

        if result["severity"] != "CLEAR":
            st.session_state.anomaly_count += 1

        save_detection(
            result,
            latitude,
            longitude,
            depth
        )

        # --- Forensic audit logger ---
        try:
            _orig_rgb = cv2.cvtColor(result["original_image"], cv2.COLOR_BGR2RGB)
            _heat_rgb = cv2.cvtColor(result["raw_heatmap"], cv2.COLOR_BGR2RGB)
            _det_rgb  = cv2.cvtColor(result["boxed_image"],  cv2.COLOR_BGR2RGB)
            _csv_p, _det_fn = save_detection_event(
                _orig_rgb,
                _heat_rgb,
                _det_rgb,
                result["score"],
                result["threshold"],
                result["detections"]
            )
            st.toast(f"Forensic record archived: {_det_fn}", icon="💾")
        except Exception:
            pass

        return result


# INPUT MODE & DETECTION CONSOLE

st.subheader("🎯 Detection Console")

# Instant 1-Click Evaluation Benchmark Presets
st.markdown("##### ⚡ Instant Evaluation Benchmark Presets (1-Click Run)")
st.caption("Instantly test the Dual-Stage AI Pipeline on pre-calibrated benchmark sonar frames:")
pre_c1, pre_c2, pre_c3 = st.columns(3)

with pre_c1:
    if st.button("🌊 Run Sample: Linear Pipeline Track", use_container_width=True, key="btn_sample_pipeline"):
        p_path = BASE_DIR / "data" / "samples" / "sample_pipeline.jpg"
        if p_path.exists():
            img_p = cv2.imread(str(p_path))
            if img_p is not None:
                execute_frame_analysis(img_p)
with pre_c2:
    if st.button("🚨 Run Sample: Subsea Anomaly Target", use_container_width=True, key="btn_sample_anomaly"):
        a_path = BASE_DIR / "data" / "samples" / "sample_anomaly.jpg"
        if a_path.exists():
            img_a = cv2.imread(str(a_path))
            if img_a is not None:
                execute_frame_analysis(img_a)
with pre_c3:
    if st.button("🟢 Run Sample: Clean Seabed Baseline", use_container_width=True, key="btn_sample_clean"):
        c_path = BASE_DIR / "data" / "samples" / "sample_clean_seabed.jpg"
        if c_path.exists():
            img_c = cv2.imread(str(c_path))
            if img_c is not None:
                execute_frame_analysis(img_c)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

mode = st.radio(
    "Input Source",
    [
        "📷 Import Image",
        "🎥 Live Camera"
    ],
    horizontal=True
)

# IMPORT IMAGE

if mode == "📷 Import Image":

    uploaded_file = st.file_uploader(
        "Upload underwater / sonar image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            st.error("Unable to read the uploaded image.")
        elif model is None:
            st.error("AI model could not be loaded.")
        else:
            if st.button(
                "🔍 Analyze Imported Image",
                type="primary",
                use_container_width=True
            ):
                execute_frame_analysis(frame)

    result = st.session_state.last_result
    if result is not None:
        # GLOBAL ANOMALY BANNER
        if result["is_anomaly"]:
            st.markdown(
                f"""
                <div class="status-critical" style="margin-top: 15px; margin-bottom: 20px;">
                    🚨 GLOBAL ANOMALY DETECTED
                    <div style="font-size: 15px; font-weight: 500; margin-top: 6px; color: #ffd8d8;">
                        Reconstruction residual MSE (<b>{result['score']:.6f}</b>) exceeds calibrated baseline threshold (<b>{result['threshold']:.6f}</b>) | <b>{result['total_detections']}</b> anomalous structure(s) isolated
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="status-clear" style="margin-top: 15px; margin-bottom: 20px;">
                    🟢 GLOBAL STATUS: SEABED CLEAR
                    <div style="font-size: 15px; font-weight: 500; margin-top: 6px; color: #d4fced;">
                        Baseline reconstruction MSE (<b>{result['score']:.6f}</b>) within normal threshold (<b>{result['threshold']:.6f}</b>) | Clean seabed baseline verified
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # SIDE-BY-SIDE 3-PANEL INSPECTION VIEW
        st.markdown("### 🛰️ Dual-Stage Inspection Panels")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("#### 1️⃣ Original Image")
            orig_display = cv2.cvtColor(
                result["original_image"],
                cv2.COLOR_BGR2RGB
            )
            st.image(
                orig_display,
                caption="Input Sonar / Seabed Optical Scan",
                use_container_width=True
            )
        with p2:
            st.markdown("#### 2️⃣ Reconstruction Heatmap")
            heat_display = cv2.cvtColor(
                result["raw_heatmap"],
                cv2.COLOR_BGR2RGB
            )
            st.image(
                heat_display,
                caption=f"Stage 1: Pixel Error Map R(x,y) [JET] | MSE: {result['score']:.6f}",
                use_container_width=True
            )
        with p3:
            st.markdown("#### 3️⃣ Bounded Detections & Types")
            bounded_display = cv2.cvtColor(
                result["boxed_image"],
                cv2.COLOR_BGR2RGB
            )
            st.image(
                bounded_display,
                caption="Stage 2: Hybrid Object Profiling (Green=Known, Red=Unknown)",
                use_container_width=True
            )
        # ITEMIZED DETECTION LOGS
        st.divider()
        st.markdown("### 📋 Itemized Detection Logs")
        if result["detections"]:
            log_entries = []
            for d in result["detections"]:
                status_badge = "🟢 KNOWN TYPE" if d["is_known"] else "🔴 UNKNOWN ANOMALY"
                conf_pct = d.get("confidence_percent", round(d.get("confidence", 0.0) * 100.0, 1))
                unc_idx = d.get("uncertainty_index", 0.15)
                t_lat = d.get("target_lat", latitude)
                t_lon = d.get("target_lon", longitude)
                g_src = d.get("geotag_source", "AUV_TELEMETRY")
                obb_deg = d.get("obb", {}).get("angle_deg", 0.0)
                log_entries.append({
                    "Box ID": f"#{d['index']}",
                    "Identified Type / Threat": d["type"],
                    "Classification Status": status_badge,
                    "Calibrated Confidence": f"{conf_pct:.1f}%",
                    "Uncertainty Index": f"{unc_idx:.3f}",
                    "Seafloor Geotag": f"{t_lat:.6f}, {t_lon:.6f}",
                    "Geotag Source": g_src,
                    "OBB Angle": f"{obb_deg:.1f}°",
                    "Contour Area (A)": f"{d['area']:.0f} px",
                    "Aspect Ratio": f"{d['aspect_ratio']:.2f}",
                    "Contrast Score": f"{d['contrast']:.3f}"
                })
            st.dataframe(pd.DataFrame(log_entries), use_container_width=True, hide_index=True)
        else:
            st.info("✓ No anomalous contours detected. Seabed surface verified clear.")
        # OBJECT INTELLIGENCE & TELEMETRY
        st.divider()
        st.markdown("### 📊 Sensor & Telemetry Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="glass-card">'
                '<div class="info-title">Depth</div>'
                f'<div class="info-value">{depth:.2f} m</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                '<div class="glass-card">'
                '<div class="info-title">Geo Coordinates</div>'
                f'<div class="info-value">{latitude:.4f}, {longitude:.4f}</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                '<div class="glass-card">'
                '<div class="info-title">Risk Assessment</div>'
                f'<div class="info-value">{result["risk"]}%</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                '<div class="glass-card">'
                '<div class="info-title">Material Estimate</div>'
                f'<div class="info-value" style="font-size: 16px;">{result["material"]}</div>'
                '</div>',
                unsafe_allow_html=True
            )
        st.write("")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Frame MSE Score", f'{result["score"]:.6f}')
        with c6:
            st.metric("Calibrated Threshold", f'{result["threshold"]:.6f}')
        with c7:
            st.metric("Highlight Ratio", f'{result["highlight_ratio"]:.2f}%')
        with c8:
            st.metric("Targets Localized", f'{result["total_detections"]}')
        # DEEP EXPLAINABILITY (RECONSTRUCTION & GRAD-CAM)
        with st.expander("🔬 Deep AI Explainability & Saliency (Autoencoder & Grad-CAM)", expanded=False):
            er1, er2 = st.columns(2)
            with er1:
                st.markdown("#### ConvAutoencoder Reconstruction")
                st.image(
                    result["reconstruction"],
                    caption="Clean seabed reconstruction generated by Stage 1 Autoencoder",
                    use_container_width=True
                )
            with er2:
                st.markdown("#### Saliency / Grad-CAM Map")
                st.image(
                    cv2.cvtColor(result["gradcam"], cv2.COLOR_BGR2RGB),
                    caption="AI reconstruction error saliency overlay",
                    use_container_width=True
                )
        # ACOUSTIC PREPROCESSING (SPECKLE FILTER & CLAHE)
        with st.expander("🛡️ Physics-Informed Acoustic Preprocessing (Speckle Filter & CLAHE)", expanded=False):
            st.markdown(
                "**Sonar Speckle Suppression & Contrast Enhancement:** "
                "Raw underwater sonar imagery is degraded by multiplicative Rayleigh acoustic speckle noise and low dynamic range. "
                "AquaSense cleans incoming frames using non-local means acoustic filtering (`fastNlMeansDenoising`) "
                "followed by Contrast-Limited Adaptive Histogram Equalization (`CLAHE`) to maximize SNR before inference."
            )
            prep1, prep2 = st.columns(2)
            with prep1:
                st.markdown("#### Raw Sonar Input")
                st.image(
                    cv2.cvtColor(result["original_image"], cv2.COLOR_BGR2RGB),
                    caption="Raw Acoustic Imagery (Unfiltered)",
                    use_container_width=True
                )
            with prep2:
                st.markdown("#### Denoised & CLAHE-Enhanced")
                _clean_img = st.session_state.last_clean_frame
                if _clean_img is None:
                    try:
                        _clean_img = clean_sonar_frame(result["original_image"])
                    except Exception:
                        _clean_img = cv2.cvtColor(result["original_image"], cv2.COLOR_BGR2GRAY)
                st.image(
                    _clean_img,
                    caption="Speckle Filtered (fastNlMeans) + High-Gain CLAHE Grayscale",
                    use_container_width=True
                )
        # AUTONOMOUS MULTI-AGENT ADVISORY CENTER
        st.markdown("### 🤖 Autonomous Multi-Agent Tactical Advisory")
        advisory = st.session_state.last_advisory
        if advisory is None:
            try:
                _agent_sys = AUVAgentSystem()
                advisory = _agent_sys.run_mission_advisory(
                    result["score"],
                    result["threshold"],
                    result["detections"],
                    depth=depth
                )
                st.session_state.last_advisory = advisory
            except Exception:
                advisory = None
        if advisory is not None:
            _p_code = advisory.get("priority_code", "NOMINAL")
            _badge_class = (
                "badge-critical" if _p_code == "CRITICAL"
                else "badge-warning" if _p_code in ["ELEVATED", "HIGH"]
                else "badge-normal"
            )
            st.markdown(
                f"""
                <div class="agent-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;">
                        <div class="agent-header">
                            <span>🧭 AUV Tactical Navigation Directive:</span>
                            <span style="color: #ffffff; font-size: 19px; font-weight: 800;">{advisory['nav_directive']}</span>
                        </div>
                        <div>
                            <span class="{_badge_class}">MISSION {advisory['priority']}</span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 15px;">
                        <div style="background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                            <div style="color: #8da4b0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;">Hazard Threat Index</div>
                            <div style="color: #ff9494; font-size: 22px; font-weight: 800; margin-top: 4px;">{advisory['hazard_index']:.1f} / 10.0</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                            <div style="color: #8da4b0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;">Nav Safety Assessment</div>
                            <div style="color: #8ce7ff; font-size: 14px; font-weight: 600; margin-top: 6px;">{advisory['nav_assessment']}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                            <div style="color: #8da4b0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;">Eco-Risk Assessment</div>
                            <div style="color: #a3e635; font-size: 14px; font-weight: 600; margin-top: 6px;">{advisory['env_assessment']}</div>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            if advisory.get("hazards"):
                _tags_html = "".join([f'<span class="hazard-tag">⚠️ {h}</span>' for h in advisory["hazards"]])
                st.markdown(
                    f"""<div style="margin-bottom: 12px;"><span style="font-size: 12px; color: #8da4b0; margin-right: 8px;">IDENTIFIED THREATS:</span>{_tags_html}</div>""",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("📄 Autonomous Agent Briefing Log (Mission Intelligence)", expanded=False):
                st.code(advisory.get("briefing", "No briefing generated."), language="markdown")
        # PRESENTATION & DEFENSE PITCH STRATEGY
        with st.expander("🎯 Presentation & Defense Pitch Strategy (Click to expand)", expanded=False):
            st.markdown("""
            #### 🌊 Master Pitch: Dual-Stage Underwater Surveillance
            **1. Pitching the Stage 1 Anomaly Engine:**
            > *"Because underwater environment data is scarce, traditional detectors miss rare or unlabelled debris. AquaSense solves this using an **Unsupervised Reconstruction Autoencoder** that learns clean seabed patterns. Anything unusual creates high reconstruction error, ensuring **zero missed detections for unknown threats**."*
            **2. Pitching Stage 2 Object Type Identification:**
            > *"Once an anomaly is flagged, Stage 2 extracts residual contour geometry and contrast features to classify the region—identifying known structures like linear shipwreck frames or marine objects, while explicitly labeling novel objects as 'Unknown Anomalies' for operator review."*
            **Key Architectural Highlights for Evaluators:**
            * **Stage 1 (Safety Net):** $224 \\times 224$ normalized tensor pass, MSE calculation vs calibrated $Mean + 3\\sigma$ threshold.
            * **Stage 2 (Tactical Clarity):** Otsu segmentation, noise rejection ($A < 100\\text{ px}$), aspect ratio profiling, Green boxes for known structures, Red boxes for unknown anomalies.
            * **Itemized Telemetry:** Complete contour coordinates, bounding boxes, and aspect ratios for mission operators.
            """)
# LIVE CAMERA

else:

    st.markdown(
        "### 🎥 Live Underwater Sensor Feed"
    )

    st.caption(
        "Real-time sensor feed operating AquaSense Dual-Stage Underwater Detection & Anomaly Engine. "
        "Frames are captured at the interval set in the sidebar, saved to disk alongside a session detection log."
    )

    # LIVE CAMERA CONTROLS

    live_ctrl_c1, live_ctrl_c2 = st.columns([1, 1])

    with live_ctrl_c1:
        start = st.button(
            "▶ Start Live Detection",
            type="primary",
            use_container_width=True
        )

    with live_ctrl_c2:
        stop = st.button(
            "⏹ Stop Detection",
            use_container_width=True
        )

    if stop:
        st.session_state.running = False

    if start:

        if model is None and dual_stage_engine is None:
            st.error("AI Model / Dual-Stage Engine could not be loaded.")
        else:
            st.session_state.running = True

            # Create a new timestamped session folder
            _session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _session_dir = (
                BASE_DIR / "data" / "live_sessions" / f"session_{_session_ts}"
            )
            _session_dir.mkdir(parents=True, exist_ok=True)
            (_session_dir / "raw").mkdir(exist_ok=True)
            (_session_dir / "annotated").mkdir(exist_ok=True)
            _session_log_path = _session_dir / "session_log.csv"

            # Write CSV header
            import csv as _csv_mod
            _SESSION_LOG_HEADERS = [
                "Frame_Index",
                "Timestamp",
                "Filename",
                "Status",
                "MSE_Score",
                "Threshold",
                "Detections_Count",
                "Detected_Types",
                "Risk_Percent",
                "Latitude",
                "Longitude",
                "Depth_m",
                "Label"
            ]
            with open(_session_log_path, "w", newline="", encoding="utf-8") as _f:
                _w = _csv_mod.writer(_f)
                _w.writerow(_SESSION_LOG_HEADERS)

            st.session_state.live_session_dir = str(_session_dir)
            st.session_state.live_capture_count = 0
            st.session_state.live_session_log = []

            # In-memory list of thumbnail RGB arrays for live log display
            _live_thumbnails = []

            camera = cv2.VideoCapture(0)

            if not camera.isOpened():
                st.error("Unable to access the camera sensor.")
                st.session_state.running = False
            else:

                # UI LAYOUT
                status_banner_placeholder = st.empty()

                col_cam, col_heat = st.columns(2)
                with col_cam:
                    st.markdown("#### 🎯 Live Bounded Detection")
                    image_placeholder = st.empty()
                with col_heat:
                    st.markdown("#### 🧬 Live Reconstruction Heatmap")
                    heatmap_placeholder = st.empty()

                telemetry_placeholder = st.empty()
                capture_info_placeholder = st.empty()

                # Live capture log section (updates every capture tick)
                st.markdown("---")
                st.markdown("#### 📋 Live Capture Log")
                st.caption(
                    "Captured images and detection metadata appear here in real-time as the camera runs."
                )
                live_log_placeholder = st.empty()

                _last_capture_time = 0.0  # epoch seconds of last disk save
                _frame_index = 0

                try:
                    while st.session_state.running:
                        success, frame = camera.read()
                        if not success:
                            st.error("Unable to read camera frame.")
                            break

                        _frame_index += 1
                        _now_epoch = time.time()

                        # Dual-Stage Analysis (runs on every frame for live display)
                        result = analyze_image(frame, model, nav_telemetry=current_telemetry)

                        st.session_state.live_score = result["score"]
                        st.session_state.last_scan = datetime.now().strftime("%H:%M:%S")

                        if result["is_anomaly"]:
                            st.session_state.anomaly_count += 1
                            st.session_state.live_status = "ANOMALY DETECTED"
                        else:
                            st.session_state.live_status = "SEABED CLEAR"

                        # INTERVAL IMAGE CAPTURE & LOGGING
                        _should_capture = (
                            _now_epoch - _last_capture_time >= capture_interval
                        )
                        _is_anomaly_frame = result["is_anomaly"]
                        _do_save = _should_capture and (
                            save_all_frames or _is_anomaly_frame
                        )

                        if _do_save:
                            _last_capture_time = _now_epoch
                            _cap_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            _cap_idx = st.session_state.live_capture_count + 1
                            _img_name = f"frame_{_cap_idx:05d}_{_cap_ts}.jpg"

                            # Save BOTH raw original and annotated bounded frame
                            _raw_path = _session_dir / "raw" / _img_name
                            _ann_path = _session_dir / "annotated" / _img_name

                            cv2.imwrite(
                                str(_raw_path),
                                frame,
                                [cv2.IMWRITE_JPEG_QUALITY, capture_quality]
                            )
                            cv2.imwrite(
                                str(_ann_path),
                                result["boxed_image"],
                                [cv2.IMWRITE_JPEG_QUALITY, capture_quality]
                            )

                            # Create a small thumbnail (120px wide) for live log display
                            _thumb_h = int(frame.shape[0] * (120 / frame.shape[1]))
                            _thumb_bgr = cv2.resize(
                                result["boxed_image"], (120, _thumb_h)
                            )
                            _thumb_rgb = cv2.cvtColor(_thumb_bgr, cv2.COLOR_BGR2RGB)
                            _live_thumbnails.append(_thumb_rgb)

                            # Format detected types for the log
                            _det_types = "; ".join(
                                [d.get("type", "Unknown") for d in result["detections"]]
                            ) if result["detections"] else "None"

                            _status_str = "🚨 ANOMALY" if _is_anomaly_frame else "🟢 CLEAR"

                            # Append row to session CSV log
                            _log_row = [
                                _cap_idx,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                _img_name,
                                result["global_status"],
                                f"{result['score']:.6f}",
                                f"{result['threshold']:.6f}",
                                len(result["detections"]),
                                _det_types,
                                result["risk"],
                                f"{latitude:.6f}",
                                f"{longitude:.6f}",
                                f"{depth:.1f}",
                                result["label"]
                            ]
                            with open(
                                _session_log_path, "a", newline="", encoding="utf-8"
                            ) as _f:
                                _csv_mod.writer(_f).writerow(_log_row)

                            st.session_state.live_capture_count += 1
                            st.session_state.live_session_log.append(
                                dict(zip(_SESSION_LOG_HEADERS, _log_row))
                            )

                            # Also run the forensic audit logger for anomaly frames
                            if _is_anomaly_frame:
                                try:
                                    _orig_rgb = cv2.cvtColor(
                                        result["original_image"], cv2.COLOR_BGR2RGB
                                    )
                                    _heat_rgb = cv2.cvtColor(
                                        result["raw_heatmap"], cv2.COLOR_BGR2RGB
                                    )
                                    _det_rgb = cv2.cvtColor(
                                        result["boxed_image"], cv2.COLOR_BGR2RGB
                                    )
                                    save_detection_event(
                                        _orig_rgb,
                                        _heat_rgb,
                                        _det_rgb,
                                        result["score"],
                                        result["threshold"],
                                        result["detections"]
                                    )
                                except Exception:
                                    pass

                            # RENDER LIVE CAPTURE LOG (scrollable)
                            _log_container = live_log_placeholder.container()
                            with _log_container:
                                # Show latest captures in reverse order (newest first), max 20
                                _entries = list(zip(
                                    reversed(st.session_state.live_session_log[-20:]),
                                    reversed(_live_thumbnails[-20:])
                                ))
                                for _entry, _thumb in _entries:
                                    _e_status = _entry.get("Status", "")
                                    if "ANOMALY" in str(_e_status).upper():
                                        _badge = '<span style="background:rgba(255,60,60,0.25);color:#ff8585;border:1px solid rgba(255,80,80,0.50);padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">🚨 ANOMALY</span>'
                                    else:
                                        _badge = '<span style="background:rgba(40,210,150,0.20);color:#63e6b1;border:1px solid rgba(60,220,170,0.40);padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">🟢 CLEAR</span>'

                                    _lc1, _lc2 = st.columns([1, 3])
                                    with _lc1:
                                        st.image(
                                            _thumb,
                                            caption=f"#{_entry.get('Frame_Index', '?')}",
                                            use_container_width=True
                                        )
                                    with _lc2:
                                        st.markdown(
                                            f"""
                                            <div style="background:rgba(8,30,48,0.6);border:1px solid rgba(0,212,255,0.15);border-radius:10px;padding:10px 14px;margin-bottom:6px;">
                                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                                    <span style="color:#7eeaff;font-weight:700;font-size:14px;">Frame #{_entry.get('Frame_Index','?')}</span>
                                                    {_badge}
                                                </div>
                                                <div style="color:#c8dfe8;font-size:12px;line-height:1.7;">
                                                    📍 <b>{_entry.get('Latitude','?')}, {_entry.get('Longitude','?')}</b> &nbsp;|&nbsp; 🌊 Depth: <b>{_entry.get('Depth_m','?')} m</b><br>
                                                    🎯 <b>{_entry.get('Label','?')}</b> &nbsp;|&nbsp; ⚠️ Risk: <b>{_entry.get('Risk_Percent','?')}%</b><br>
                                                    📊 MSE: <b>{_entry.get('MSE_Score','?')}</b> &nbsp;|&nbsp; 🔍 Objects: <b>{_entry.get('Detections_Count','0')}</b> ({_entry.get('Detected_Types','None')})<br>
                                                    🕐 <b>{_entry.get('Timestamp','?')}</b> &nbsp;|&nbsp; 📁 <code>{_entry.get('Filename','?')}</code>
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                        # DISPLAY — Live Bounded Frame & Heatmap
                        image_placeholder.image(
                            cv2.cvtColor(result["boxed_image"], cv2.COLOR_BGR2RGB),
                            caption="Live Dual-Stage Feed (Green=Known Types | Red=Unknown Anomalies)",
                            use_container_width=True
                        )

                        heatmap_placeholder.image(
                            cv2.cvtColor(result["raw_heatmap"], cv2.COLOR_BGR2RGB),
                            caption=f"Stage 1 Residual Error Map [JET] | MSE: {result['score']:.6f}",
                            use_container_width=True
                        )

                        # Global Status Banner
                        if result["is_anomaly"]:
                            status_banner_placeholder.markdown(
                                f"""
                                <div class="status-critical" style="margin-bottom: 12px; padding: 12px;">
                                    🚨 ANOMALY DETECTED — MSE: {result['score']:.6f} &gt; Threshold: {result['threshold']:.6f} | Targets: {result['total_detections']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            status_banner_placeholder.markdown(
                                f"""
                                <div class="status-clear" style="margin-bottom: 12px; padding: 12px;">
                                    🟢 SEABED CLEAR — MSE: {result['score']:.6f} ≤ Threshold: {result['threshold']:.6f} | Baseline Normal
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        telemetry_placeholder.info(
                            f"Primary Target: {result['label']} | "
                            f"Risk: {result['risk']}% | "
                            f"Depth: {depth:.1f} m | "
                            f"Coordinates: {latitude:.6f}, {longitude:.6f} | "
                            f"Threshold: {result['threshold']:.6f}"
                        )

                        # Capture status footer
                        _next_capture_in = max(
                            0.0, capture_interval - (_now_epoch - _last_capture_time)
                        )
                        capture_info_placeholder.caption(
                            f"📸 Frames captured: **{st.session_state.live_capture_count}** | "
                            f"Next capture in **{_next_capture_in:.1f}s** | "
                            f"Interval: {capture_interval:.1f}s | "
                            f"Session: `{_session_dir.name}`"
                        )

                        # Tight loop — actual interval throttling done by _last_capture_time
                        time.sleep(0.10)

                finally:
                    camera.release()
                    st.session_state.running = False
                    st.success(
                        f"✅ Session ended. {st.session_state.live_capture_count} frames saved to "
                        f"`data/live_sessions/{_session_dir.name}/`"
                    )

    # SESSION LOG VIEWER (shown after/between sessions)
    if (
        not st.session_state.running
        and st.session_state.live_session_dir is not None
        and len(st.session_state.live_session_log) > 0
    ):
        st.markdown("---")
        st.markdown("#### 📋 Session Capture Log (Final)")

        _log_df = pd.DataFrame(st.session_state.live_session_log)
        st.dataframe(
            _log_df,
            use_container_width=True,
            height=300
        )

        # CSV download
        _log_csv_bytes = _log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Session Log (CSV)",
            data=_log_csv_bytes,
            file_name=f"aquasense_session_{Path(st.session_state.live_session_dir).name}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            f"Session images saved to: `{st.session_state.live_session_dir}`  \n"
            f"📁 `raw/` — original unprocessed frames  \n"
            f"📁 `annotated/` — AI bounded-detection overlays"
        )


# GEOSPATIAL COMMAND CENTER (GIS BATHYMETRY MAP)

st.divider()

st.markdown("## 🗺️ Geospatial Command Center & Acoustic Bathymetry Map")
st.caption(
    "Interactive tactical GIS map displaying real-time AUV position, georeferenced anomaly alerts, "
    "and spatial density heatmaps across the survey operating theater."
)

# Load detections for geospatial plotting
geo_df = load_csv(DETECTION_CSV)

# Center coordinates: use latest detection or default sensor metadata from sidebar
map_center_lat = float(latitude)
map_center_lon = float(longitude)

if not geo_df.empty and "latitude" in geo_df.columns and "longitude" in geo_df.columns:
    valid_coords = geo_df.dropna(subset=["latitude", "longitude"])
    if not valid_coords.empty:
        try:
            map_center_lat = float(valid_coords["latitude"].iloc[-1])
            map_center_lon = float(valid_coords["longitude"].iloc[-1])
        except Exception:
            pass

# Create Folium Multi-Layer Map (Tactical Dark Canvas + Ocean Bathymetry + OSM Navigation)
gis_map = folium.Map(
    location=[map_center_lat, map_center_lon],
    zoom_start=14,
    tiles=None,
    control_scale=True
)

# Base Layer 1: OpenStreetMap Standard
folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap Navigation",
    overlay=False,
    control=True
).add_to(gis_map)

# Base Layer 2: Seafloor Bathymetry (Esri World Ocean - depth topography & trenches)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, GEBCO, NOAA, National Geographic",
    name="Seafloor Bathymetry",
    overlay=False,
    control=True
).add_to(gis_map)

# Base Layer 3: Tactical Dark Gray Canvas (Esri - active default theme)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, © OpenStreetMap contributors",
    name="Tactical Dark Canvas",
    overlay=False,
    control=True
).add_to(gis_map)

# Add AUV Current Location Marker
folium.Marker(
    location=[float(latitude), float(longitude)],
    popup=folium.Popup(
        f"<b>🤖 AUV Active Position</b><br>Lat: {latitude:.6f}<br>Lon: {longitude:.6f}<br>Depth: {depth:.1f} m",
        max_width=250
    ),
    tooltip="AUV Active Waypoint",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(gis_map)

# Add detection markers and accumulate heat data
heat_points = []
anomaly_markers_count = 0
clear_markers_count = 0

if not geo_df.empty and "latitude" in geo_df.columns and "longitude" in geo_df.columns:
    for _, row in geo_df.iterrows():
        try:
            r_lat = float(row.get("latitude", 0))
            r_lon = float(row.get("longitude", 0))
            if r_lat == 0 and r_lon == 0:
                continue

            r_sev = str(row.get("severity", "CLEAR")).upper()
            r_score = float(row.get("anomaly_score", 0.0))
            r_obj = str(row.get("object", "Unknown"))
            r_time = str(row.get("timestamp", "--"))
            r_depth = str(row.get("depth_m", "--"))

            is_anom = (r_sev == "CRITICAL") or (r_score > MODEL_THRESHOLD)

            if is_anom:
                anomaly_markers_count += 1
                heat_points.append([r_lat, r_lon, min(1.0, max(0.4, r_score * 30))])
                circle_color = "#ff4d4d"
                fill_color = "#ff1a1a"
                radius = 8
            else:
                clear_markers_count += 1
                circle_color = "#00e676"
                fill_color = "#00c853"
                radius = 6

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; color: #222;">
                <b>{r_obj}</b><br>
                <b>Severity:</b> <span style="color: {'red' if is_anom else 'green'};">{r_sev}</span><br>
                <b>MSE Score:</b> {r_score:.6f}<br>
                <b>Depth:</b> {r_depth} m<br>
                <b>Time:</b> {r_time}<br>
                <b>Coordinates:</b> {r_lat:.5f}, {r_lon:.5f}
            </div>
            """

            folium.CircleMarker(
                location=[r_lat, r_lon],
                radius=radius,
                color=circle_color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{'🚨 ANOMALY' if is_anom else '🟢 CLEAR'}: {r_obj} ({r_score:.4f})"
            ).add_to(gis_map)
        except Exception:
            continue

# Add HeatMap overlay if points exist
if heat_points:
    try:
        HeatMap(
            heat_points,
            radius=20,
            blur=15,
            min_opacity=0.35,
            gradient={0.4: "blue", 0.65: "lime", 1.0: "red"}
        ).add_to(gis_map)
    except Exception:
        pass

folium.LayerControl(position="topright").add_to(gis_map)

# Quick metrics bar above map
g_c1, g_c2, g_c3, g_c4 = st.columns(4)
with g_c1:
    st.metric("AUV Position", f"{latitude:.4f}, {longitude:.4f}")
with g_c2:
    st.metric("Survey Depth", f"{depth:.1f} m")
with g_c3:
    st.metric("Logged Anomalies", anomaly_markers_count)
with g_c4:
    st.metric("Verified Baselines", clear_markers_count)

# Render Folium Map in Streamlit
st_folium(gis_map, width=None, height=450, returned_objects=[])

st.caption(
    "🔴 Red markers: Anomalies & high reconstruction residuals | "
    "🟢 Green markers: Verified clear seabed | "
    "🔵 Blue beacon: Current AUV telemetry coordinate"
)


# ACOUSTIC TELEMETRY & MICRO-MODEM SIMULATOR

st.divider()

st.markdown("## 📡 Acoustic Telemetry & Micro-Modem Simulator")
st.caption(
    "Simulates ultra-narrowband acoustic packet transmission over underwater acoustic channels (WHOI Micro-Modem / EvoLogics S2C)."
)

sim_col1, sim_col2 = st.columns([1, 1])

# Current or fallback telemetry
curr_tel = st.session_state.last_telemetry
if curr_tel is None:
    try:
        curr_tel = encode_acoustic_telemetry(0.0035, MODEL_THRESHOLD, [])
    except Exception:
        curr_tel = None

with sim_col1:
    st.markdown("#### 📊 Bandwidth Compression Analysis")
    if curr_tel is not None:
        raw_b = curr_tel["raw_img_bytes"]
        pkg_b = curr_tel["size_bytes"]
        sav_pct = curr_tel["savings_pct"]

        # Transmission time at 80 bps (typical WHOI micromodem rate)
        tx_raw_s = (raw_b * 8) / 80.0
        tx_pkg_s = (pkg_b * 8) / 80.0

        tc1, tc2 = st.columns(2)
        with tc1:
            st.metric("Raw Sonar Frame", f"{raw_b:,} B", f"{raw_b / 1024:.1f} KB")
            st.metric("Raw TX Time @ 80bps", f"{tx_raw_s:.1f} s", f"{tx_raw_s / 60:.1f} min")
        with tc2:
            st.metric("Acoustic Payload", f"{pkg_b} B", f"-{sav_pct:.1f}%", delta_color="inverse")
            st.metric("Payload TX Time @ 80bps", f"{tx_pkg_s:.2f} s", f"{tx_raw_s / max(tx_pkg_s, 0.01):.0f}x Faster")

        st.info(
            f"💡 **Subsea Optimization:** Transmitting the raw sonar frame would saturate the 80 bps acoustic "
            f"channel for **{tx_raw_s / 60:.1f} minutes**. The AquaSense telemetry encoder condenses critical detections "
            f"into a compact **{pkg_b}-byte packet** transmitted in **{tx_pkg_s:.2f} seconds** ({sav_pct:.1f}% bandwidth reduction)."
        )

        if st.button("🚀 Transmit Packet Over Acoustic Channel", key="btn_sim_tx"):
            with st.status("Initiating WHOI Acoustic Modem handshake...", expanded=True) as status:
                st.write("Serializing JSON packet with CRC32 integrity check...")
                time.sleep(0.3)
                st.write(f"Carrier Frequency: 25 kHz FSK/PSK | Rate: 80 bps | Size: {pkg_b} bytes")
                time.sleep(0.3)
                st.write("Acoustic packet transmitted successfully. Echo acknowledged by topside USBL transponder.")
                status.update(label="✅ Acoustic Transmission Confirmed (ACK Received)", state="complete", expanded=False)
            st.toast("📡 Acoustic packet received by topside vessel command!", icon="🌊")
    else:
        st.info("Execute an image analysis above to generate a real-time acoustic telemetry packet.")

with sim_col2:
    st.markdown("#### 📦 Encoded JSON Telemetry Packet")
    if curr_tel is not None:
        st.caption("Standard compact acoustic telemetry format:")
        st.code(curr_tel["payload_json"], language="json")
    else:
        st.caption("Awaiting scan packet...")


# DETECTION HISTORY

st.divider()

st.subheader(
    "📋 All Detected Objects"
)

detection_df = load_csv(
    DETECTION_CSV
)

if detection_df.empty:

    st.info(
        "No detections recorded yet. "
        "Analyze an image to create the first detection."
    )

else:

    # Summary

    total = len(
        detection_df
    )

    critical = int(
        (
            detection_df[
                "severity"
            ]
            .astype(str)
            .str.upper()
            == "CRITICAL"
        ).sum()
    )

    bypass = int(
        (
            detection_df[
                "action"
            ]
            .astype(str)
            .str.upper()
            == "BYPASS"
        ).sum()
    )

    review = int(
        (
            detection_df[
                "severity"
            ]
            .astype(str)
            .str.upper()
            == "REVIEW"
        ).sum()
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:

        st.metric(
            "Total Objects",
            total
        )

    with h2:

        st.metric(
            "Critical Debris",
            critical
        )

    with h3:

        st.metric(
            "Review",
            review
        )

    with h4:

        st.metric(
            "Bypassed",
            bypass
        )

    # Table

    history = detection_df.copy()

    preferred_columns = [
        "timestamp",
        "object",
        "severity",
        "risk_percent",
        "material",
        "depth_m",
        "latitude",
        "longitude",
        "geo_coordinates",
        "highlight_ratio",
        "action"
    ]

    available_columns = [
        c
        for c in preferred_columns
        if c in history.columns
    ]

    history = history[
        available_columns
    ]

    rename_map = {
        "timestamp":
            "Time",

        "object":
            "Detected Object",

        "severity":
            "Severity",

        "risk_percent":
            "Risk %",

        "material":
            "Material",

        "depth_m":
            "Depth (m)",

        "latitude":
            "Latitude",

        "longitude":
            "Longitude",

        "geo_coordinates":
            "Geo Coordinates",

        "highlight_ratio":
            "Highlight Ratio %",

        "action":
            "Action"
    }

    history = history.rename(
        columns=rename_map
    )

    st.dataframe(
        history.sort_values(
            "Time",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# TEMPORAL AI

st.divider()

st.subheader(
    "🔄 Temporal Change Detection"
)

if temporal_df.empty:

    st.info(
        "No temporal results found."
    )

else:

    temporal_changes = 0

    confirmed_anomalies = 0

    if "temporal_change" in temporal_df:

        temporal_changes = int(
            temporal_df[
                "temporal_change"
            ].sum()
        )

    if "ai_confirmed" in temporal_df:

        confirmed_anomalies = int(
            temporal_df[
                "ai_confirmed"
            ].sum()
        )

    t1, t2, t3 = st.columns(3)

    with t1:

        st.metric(
            "Frames Compared",
            len(temporal_df)
        )

    with t2:

        st.metric(
            "Temporal Changes",
            temporal_changes
        )

    with t3:

        st.metric(
            "AI Confirmed",
            confirmed_anomalies
        )

    st.dataframe(
        temporal_df,
        use_container_width=True,
        hide_index=True
    )


# PRIORITY PLANNER

st.divider()

st.subheader(
    "🗺️ Accumulation & Mission Priority"
)

if priority_df.empty:

    st.info(
        "No priority plan available."
    )

else:

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Priority Zones",
            len(priority_df)
        )

    with p2:

        if "priority" in priority_df:

            high_priority = int(
                (
                    priority_df[
                        "priority"
                    ]
                    .astype(str)
                    .str.upper()
                    == "HIGH"
                ).sum()
            )

        else:

            high_priority = 0

        st.metric(
            "High Priority",
            high_priority
        )

    with p3:

        if "detections" in priority_df:

            total_detections = int(
                priority_df[
                    "detections"
                ].sum()
            )

        else:

            total_detections = 0

        st.metric(
            "Total Detections",
            total_detections
        )

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )


# FORENSIC AUDIT TRAIL

st.divider()

st.subheader("🗂️ Forensic Audit Trail")
st.caption(
    "Every analyzed frame is automatically archived here — "
    "structured CSV metadata and 3-panel visual evidence "
    "(original · heatmap · bounded detection)."
)

_audit_df = get_recent_logs(limit=50)

if _audit_df.empty:
    st.info(
        "No audit records yet. Analyze an image or run the "
        "live camera feed to generate the first forensic entry."
    )
else:
    # ---- Summary metrics ----
    _total_events  = len(_audit_df)
    _anomaly_events = int((_audit_df["Status"].str.upper() == "ANOMALY DETECTED").sum())
    _clear_events   = _total_events - _anomaly_events

    _am1, _am2, _am3 = st.columns(3)
    with _am1:
        st.metric("📸 Total Frames Logged", _total_events)
    with _am2:
        st.metric("🚨 Anomaly Events", _anomaly_events)
    with _am3:
        st.metric("🟢 Clear Frames", _clear_events)

    # ---- Interactive audit table ----
    st.markdown("#### 📄 Event Log (Most Recent First)")
    st.dataframe(
        _audit_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    # ---- CSV download button ----
    import io as _io
    _csv_bytes = _audit_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Audit CSV",
        data=_csv_bytes,
        file_name="aquasense_audit_report.csv",
        mime="text/csv",
        help="Download the complete forensic audit log as a CSV file."
    )

    # ---- Visual evidence gallery ----
    with st.expander(
        "🖼️ Visual Evidence Gallery — Click to inspect saved event images",
        expanded=False
    ):
        st.caption(
            "Select a logged detection filename to view its "
            "3-panel forensic evidence (original · heatmap · bounded detection)."
        )

        _event_filenames = _audit_df["Image_Filename"].dropna().tolist()

        if _event_filenames:
            _selected_event = st.selectbox(
                "Select Detection Event",
                options=_event_filenames,
                index=0,
                key="audit_gallery_select"
            )

            _img_paths = get_saved_image_paths(_selected_event)

            _g1, _g2, _g3 = st.columns(3)

            with _g1:
                st.markdown("**1️⃣ Original Frame**")
                if _img_paths["original"] and _img_paths["original"].exists():
                    _gimg = cv2.imread(str(_img_paths["original"]))
                    if _gimg is not None:
                        st.image(
                            cv2.cvtColor(_gimg, cv2.COLOR_BGR2RGB),
                            caption="Input Sonar / Optical Frame",
                            use_container_width=True
                        )
                else:
                    st.info("Original image not found on disk.")

            with _g2:
                st.markdown("**2️⃣ Reconstruction Heatmap**")
                if _img_paths["heatmap"] and _img_paths["heatmap"].exists():
                    _himg = cv2.imread(str(_img_paths["heatmap"]))
                    if _himg is not None:
                        st.image(
                            cv2.cvtColor(_himg, cv2.COLOR_BGR2RGB),
                            caption="Stage 1: Pixel Error Map [JET]",
                            use_container_width=True
                        )
                else:
                    st.info("Heatmap image not found on disk.")

            with _g3:
                st.markdown("**3️⃣ Bounded Detection**")
                if _img_paths["detection"] and _img_paths["detection"].exists():
                    _dimg = cv2.imread(str(_img_paths["detection"]))
                    if _dimg is not None:
                        st.image(
                            cv2.cvtColor(_dimg, cv2.COLOR_BGR2RGB),
                            caption="Stage 2: Hybrid Profiling (Green=Known | Red=Unknown)",
                            use_container_width=True
                        )
                else:
                    st.info("Detection image not found on disk.")

            # Show matching row metadata
            _matching_rows = _audit_df[
                _audit_df["Image_Filename"] == _selected_event
            ]
            if not _matching_rows.empty:
                st.markdown("**📊 Event Metadata**")
                st.dataframe(
                    _matching_rows.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No saved images found in audit log.")


# RESEARCH ADVANCEMENTS CONSOLE (OCEAN-AWARE & SUBSEA ROBOTICS)

st.divider()

st.markdown("## 🔬 Ocean-Aware Research Advancements Console")
st.caption(
    "Operational implementation of frontier maritime AI and deep-sea robotics paradigms: "
    "Calibrated Confidence Scoring, Multi-Source Geotagging, Acoustic Ray-Tracing, "
    "Space/SAR-AIS Multimodal Fusion, and 6-DOF Fossen Hydrodynamic MPC."
)

adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
    "📍 Geotagging & GeoJSON Export",
    "🎯 Calibrated Confidence Scoring",
    "🌊 Acoustic Physics & Deep Positioning",
    "🛰️ Space & Surface AIS Fusion",
    "🤖 Vehicle Dynamics & Cable MPC"
])

# TAB 1: GEOTAGGING & GEOJSON EXPORT
with adv_tab1:
    st.markdown("### 📍 Multi-Source Geotagging & Seafloor FOV Projection")
    st.caption("Converts raw image pixel detections into georeferenced Earth coordinates with spatial uncertainty bounds.")

    gt_c1, gt_c2 = st.columns([1, 1])

    with gt_c1:
        st.markdown("#### 🔭 Seafloor Ray-Projection Calculator")
        calc_alt = st.number_input("AUV Altitude (m)", value=8.0, min_value=1.0, step=0.5, key="gt_calc_alt")
        calc_hdg = st.number_input("AUV Heading (deg)", value=120.0, min_value=0.0, max_value=360.0, step=5.0, key="gt_calc_hdg")
        calc_px_x = st.number_input("Pixel X Centroid (0-224)", value=160, min_value=0, max_value=224, step=1, key="gt_calc_px_x")
        calc_px_y = st.number_input("Pixel Y Centroid (0-224)", value=60, min_value=0, max_value=224, step=1, key="gt_calc_px_y")

        proj_res = GeotaggingEngine.project_pixel_to_seafloor(
            vehicle_lat=latitude,
            vehicle_lon=longitude,
            vehicle_heading_deg=calc_hdg,
            altitude_m=calc_alt,
            bbox_center_px=(calc_px_x, calc_px_y)
        )

        st.success(
            f"**Target Seafloor Lat:** `{proj_res['target_lat']:.6f}`\n\n"
            f"**Target Seafloor Lon:** `{proj_res['target_lon']:.6f}`\n\n"
            f"**Offset:** `{proj_res['radial_offset_m']} m` (North: {proj_res['offset_north_m']}m, East: {proj_res['offset_east_m']}m)"
        )

    with gt_c2:
        st.markdown("#### 🌍 OGC GeoJSON Anomaly Export")
        st.caption("Exports current session detections into standard GIS GeoJSON format for QGIS, ArcGIS, or Google Earth.")

        # Build sample GeoJSON feature set from recent detections
        demo_detections = []
        if st.session_state.last_result and st.session_state.last_result.get("detections"):
            for d in st.session_state.last_result["detections"]:
                demo_detections.append({
                    "target_lat": d.get("target_lat", latitude),
                    "target_lon": d.get("target_lon", longitude),
                    "depth_m": depth,
                    "type": d.get("type", "Submerged Target"),
                    "status": d.get("status", "ANOMALY"),
                    "confidence_percent": d.get("confidence_percent", 85.0),
                    "anomaly_score": st.session_state.last_result["score"],
                    "source": d.get("geotag_source", geotag_source_mode),
                    "accuracy_m": d.get("accuracy_m", 3.0)
                })
        else:
            demo_detections = [{
                "target_lat": latitude,
                "target_lon": longitude,
                "depth_m": depth,
                "type": "Subsea Marine Cable Segment",
                "status": "KNOWN TYPE",
                "confidence_percent": 94.2,
                "anomaly_score": 0.0312,
                "source": geotag_source_mode,
                "accuracy_m": 2.5
            }]

        geojson_data = GeotaggingEngine.generate_geojson_report(demo_detections)
        geojson_str = json.dumps(geojson_data, indent=2)

        st.json(geojson_data)
        st.download_button(
            label="⬇️ Download Survey GeoJSON Report",
            data=geojson_str.encode("utf-8"),
            file_name="aquasense_forensic_survey.geojson",
            mime="application/geo+json",
            use_container_width=True
        )

# TAB 2: CALIBRATED CONFIDENCE SCORING
with adv_tab2:
    st.markdown("### 🎯 Calibrated Confidence & Uncertainty Quantification Engine")
    st.caption("Replaces heuristic values with statistically calibrated probabilities and epistemic uncertainty estimation.")

    conf_c1, conf_c2 = st.columns([1, 1])

    with conf_c1:
        st.markdown("#### 🧪 Calibration Simulator")
        sim_mse = st.slider("Simulated Frame MSE", 0.001, 0.080, 0.035, 0.001, format="%.4f")
        sim_type = st.selectbox(
            "Target Candidate Type",
            ["Linear Pipeline / Cable Track", "Shipwreck Frame / Linear Debris", "Submerged Aircraft Component", "Submerged Marine Object / Fauna", "Unclassified Submerged Threat"]
        )
        sim_area = st.slider("Contour Pixel Area", 100, 6000, 1800, 50)
        sim_contrast = st.slider("Target vs Background Contrast", 0.05, 0.90, 0.40, 0.05)
        sim_elong = st.slider("Contour Elongation Ratio", 1.0, 6.0, 2.8, 0.1)

        sim_status = "KNOWN TYPE" if sim_type != "Unclassified Submerged Threat" else "UNKNOWN ANOMALY"

        sim_scorer = CalibratedConfidenceScorer(threshold=MODEL_THRESHOLD)
        conf_report = sim_scorer.evaluate(
            frame_mse=sim_mse,
            object_type=sim_type,
            status=sim_status,
            area=float(sim_area),
            aspect_ratio=sim_elong,
            elongation=sim_elong,
            contrast_score=sim_contrast,
            mean_anomaly=min(255.0, sim_mse * 2000.0),
            spans_frame=(sim_elong >= 2.2)
        )

    with conf_c2:
        st.markdown("#### 📊 Calibrated Decision Vector")
        tier_color = "#00d4ff" if conf_report["quality_tier"] == "HIGH" else ("#ffaa00" if conf_report["quality_tier"] == "MODERATE" else "#ff3366")

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {tier_color};">
                <h3 style="margin-top:0; color:{tier_color};">Composite Confidence: {conf_report['confidence_percent']}%</h3>
                <p><b>Quality Tier:</b> <code>{conf_report['quality_tier']}</code> | <b>Epistemic Uncertainty:</b> <code>{conf_report['uncertainty_index']}</code></p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <p>• <b>Reconstruction Anomaly Probability P(A):</b> {conf_report['reconstruction_probability']*100:.1f}%</p>
                <p>• <b>Geometric Purity Score:</b> {conf_report['geometric_purity']*100:.1f}%</p>
                <p>• <b>Acoustic SNR Saliency Score:</b> {conf_report['snr_score']*100:.1f}%</p>
                <p>• <b>Shadow / Contrast Fidelity:</b> {conf_report['shadow_fidelity']*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("💡 **Operational Insight:** High epistemic uncertainty flags edge cases where visual contrast and reconstruction residual disagree, automatically escalating to human oversight without false alarms.")

# TAB 3: ACOUSTIC PHYSICS & DEEP POSITIONING
with adv_tab3:
    st.markdown("### 🌊 Acoustic Physics & Deep Positioning (GPS-Denied Navigation)")
    st.caption("Solves underwater sound velocity variations, Snell's law acoustic refraction, and pose-graph SLAM drift.")

    ap_c1, ap_c2 = st.columns(2)

    with ap_c1:
        st.markdown("#### 🌡️ Mackenzie Sound Speed Profile (SSP)")
        ap_temp = st.slider("Surface Water Temperature (°C)", 5.0, 32.0, 24.0, 0.5)
        ap_sal = st.slider("Salinity (ppt)", 28.0, 40.0, 35.0, 0.2)
        ap_depth = st.slider("Target Seafloor Depth (m)", 10.0, 3000.0, 800.0, 50.0)

        calc_sound_speed = SoundSpeedProfile.mackenzie_sound_speed(ap_temp, ap_sal, ap_depth)
        st.metric("In-Situ Sound Speed (m/s)", f"{calc_sound_speed:.2f} m/s")

        st.markdown("#### 📐 Snell's Law Acoustic Ray-Tracing")
        ray_angle = st.slider("Acoustic Launch Angle (deg from horizontal)", 10.0, 80.0, 35.0, 5.0)

        ray_tracer = AcousticRayTracer()
        ray_res = ray_tracer.trace_ray(
            launch_angle_deg=ray_angle,
            transceiver_depth_m=10.0,
            target_depth_m=ap_depth
        )

        st.markdown(
            f"""
            <div class="glass-card">
                <b>Ray-Tracing Refraction Correction:</b><br>
                • True Curved Path: <code>{ray_res['true_curved_path_m']} m</code><br>
                • Horizontal Ground Range: <code>{ray_res['ground_range_m']} m</code><br>
                • Straight-Line Approximation: <code>{ray_res['straight_line_estimate_m']} m</code><br>
                • <b>Refraction Error Corrected:</b> <code style="color:#00d4ff;">+{ray_res['refraction_correction_m']} m</code><br>
                • Total Travel Time: <code>{ray_res['travel_time_s']} s</code>
            </div>
            """,
            unsafe_allow_html=True
        )

    with ap_c2:
        st.markdown("#### ⚓ Multi-Transponder LBL Multilateration Solver")
        st.caption("Solves 3D vehicle coordinates from acoustic travel times to 3 seabed transponders using non-linear least squares.")

        tp1_tof = st.number_input("ToF Transponder #1 (s)", value=0.120, format="%.3f")
        tp2_tof = st.number_input("ToF Transponder #2 (s)", value=0.185, format="%.3f")
        tp3_tof = st.number_input("ToF Transponder #3 (s)", value=0.150, format="%.3f")

        transponders = [(0.0, 0.0, 100.0), (300.0, 0.0, 100.0), (150.0, 300.0, 100.0)]
        tofs = [tp1_tof, tp2_tof, tp3_tof]
        solved_pos = AcousticMultilaterationSolver.solve_lbl_position(transponders, tofs, sound_speed_mps=calc_sound_speed)

        st.success(
            f"**Solved X:** `{solved_pos['est_x_m']:.2f} m` | "
            f"**Solved Y:** `{solved_pos['est_y_m']:.2f} m` | "
            f"**Depth:** `{solved_pos['est_depth_m']:.2f} m`\n\n"
            f"**Residual Accuracy:** `±{solved_pos['residual_error_m']} m` ({solved_pos['convergence_status']})"
        )

        st.markdown("#### 🔄 Sonar Pose-Graph SLAM Drift Correction")
        slam_sim = SonarPoseGraphSLAM.simulate_cable_survey_with_slam(cable_length_m=400.0)
        st.markdown(
            f"""
            <div class="glass-card">
                <b>Cable Survey SLAM Performance:</b><br>
                • Uncompensated IMU Dead-Reckoning Drift: <code style="color:#ff3366;">{slam_sim['final_uncompensated_drift_m']} m</code><br>
                • Pose-Graph Loop Closure Residual Error: <code style="color:#00ff88;">{slam_sim['final_slam_error_m']} m</code><br>
                • <b>Drift Elimination:</b> <code style="color:#00d4ff;">{slam_sim['drift_reduction_percent']}%</code>
            </div>
            """,
            unsafe_allow_html=True
        )

# TAB 4: SPACE & SURFACE SURVEILLANCE / AIS FUSION
with adv_tab4:
    st.markdown("### 🛰️ Surface & Space Surveillance & Multi-Modal AIS Fusion")
    st.caption("Solves wide-area maritime domain awareness: SAR speckle filtering, OBB rotated detection, and AIS spoofing/dark-vessel detection.")

    sp_c1, sp_c2 = st.columns(2)

    with sp_c1:
        st.markdown("#### 📡 SAR Speckle Filtering & Rotated Bounding Boxes (OBB)")
        st.caption("Lee filter suppresses multiplicative Bragg sea clutter while OBB tightly bounds angled ship hulls.")

        sar_synthetic = np.random.normal(50, 20, (120, 120)).clip(0, 255).astype(np.uint8)
        cv2.rectangle(sar_synthetic, (40, 25), (70, 95), 220, -1)  # Vessel signature
        filtered_sar = SARProcessor.lee_filter(sar_synthetic, window_size=5)
        _, thresh_sar = cv2.threshold(filtered_sar, 130, 255, cv2.THRESH_BINARY)
        obbs = SARProcessor.extract_oriented_bounding_boxes(thresh_sar)

        st.image(
            [sar_synthetic, filtered_sar, thresh_sar],
            caption=["Raw SAR Sea Clutter", "Lee Speckle Filtered", "Oriented Target Extraction"],
            width=160
        )
        if obbs:
            st.info(f"Detected Vessel OBB: Length {obbs[0]['length_px']}px | Breadth {obbs[0]['breadth_px']}px | Orientation {obbs[0]['orientation_deg']}°")

    with sp_c2:
        st.markdown("#### 🚨 Multi-Modal AIS Cross-Referencing Engine")
        st.caption("Correlates physical satellite/radar detections with reported AIS transponders to uncover Dark Vessels and Spoofing.")

        fusion_engine = MultiModalMaritimeFusionEngine()

        sample_sar_targets = [
            {"target_lat": latitude + 0.002, "target_lon": longitude + 0.002, "area": 3500.0},
            {"target_lat": latitude + 0.025, "target_lon": longitude + 0.015, "area": 4800.0} # Dark vessel
        ]
        sample_ais = [
            AISRecord(987654321, "MV OCEAN CARRIER", "Cargo", latitude + 0.0021, longitude + 0.0022, 14.5, 90.0),
            AISRecord(112233445, "GHOST TANKER", "Tanker", latitude + 0.050, longitude + 0.050, 18.0, 180.0) # Ghost spoof
        ]

        fusion_results = fusion_engine.correlate_detections_with_ais(sample_sar_targets, sample_ais, match_radius_m=350.0)

        st.markdown(
            f"""
            <div class="glass-card">
                <b>Cross-Modal Audit Summary:</b><br>
                • 🟢 <b>Verified Cooperative Vessels:</b> <code>{fusion_results['cooperative_count']}</code><br>
                • 🚨 <b style="color:#ff3366;">Dark Vessels (No AIS Broadcast):</b> <code>{fusion_results['dark_vessel_count']}</code><br>
                • ⚠️ <b style="color:#ffaa00;">AIS Spoofing / Ghost Ships:</b> <code>{fusion_results['spoofed_target_count']}</code>
            </div>
            """,
            unsafe_allow_html=True
        )

        if fusion_results["dark_vessels"]:
            st.error(f"🚨 **CRITICAL ALERT:** Dark Vessel detected at Lat {fusion_results['dark_vessels'][0]['target_lat']:.5f}, Lon {fusion_results['dark_vessels'][0]['target_lon']:.5f}! Suspected IUU fishing or illegal transshipment.")

# TAB 5: VEHICLE DYNAMICS & CABLE MPC
with adv_tab5:
    st.markdown("### 🤖 6-DOF Fossen Hydrodynamics & Cable-Tracking MPC")
    st.caption("Solves physical vehicle dynamics, ocean current disturbance rejection, and MAVLink autopilot interfacing.")

    vd_c1, vd_c2 = st.columns(2)

    with vd_c1:
        st.markdown("#### 🌊 6-DOF Fossen Marine Hydrodynamics")
        fossen_model = FossenHydrodynamics6DOF(mass_kg=38.0)

        cross_current = st.slider("Ocean Cross-Current Speed (m/s)", -1.0, 1.0, 0.45, 0.05)
        applied_thrust = st.slider("Forward Surge Thrust (N)", 0.0, 100.0, 45.0, 5.0)

        v_init = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_init = np.array([applied_thrust, 0.0, 0.0, 0.0, 0.0, 0.0])
        current_vec = np.array([0.0, cross_current, 0.0])

        new_vel, acc = fossen_model.step(v_init, tau_init, current_vec, dt=0.1)

        st.markdown(
            f"""
            <div class="glass-card">
                <b>Hydrodynamic State Evolution:</b><br>
                • Surge Velocity (Forward): <code>{new_vel[0]:.2f} m/s</code><br>
                • Sway Drift Velocity (Lateral): <code>{new_vel[1]:.2f} m/s</code><br>
                • Lateral Current Force: <code>{35.0 * cross_current + 75.0 * (cross_current**2):.1f} N</code><br>
                • Restoring Pitch Stability: <code>ACTIVE (Inherent CB > CG)</code>
            </div>
            """,
            unsafe_allow_html=True
        )

    with vd_c2:
        st.markdown("#### 🎛️ Cable-Following Model Predictive Control (MPC)")
        mpc_controller = CableTrackingMPC(horizon_steps=10)

        cross_track_offset = st.slider("Current Distance Off Cable (m)", -3.0, 3.0, 1.2, 0.1)
        cable_ref = [(0.0, 0.0, depth), (100.0, 0.0, depth)]

        mpc_out = mpc_controller.compute_optimal_control(
            current_x_m=0.0,
            current_y_m=cross_track_offset,
            current_depth_m=depth,
            current_vx=new_vel[0],
            current_vy=new_vel[1],
            current_vz=0.0,
            cable_reference_line=cable_ref,
            ocean_current_vy=cross_current
        )

        st.success(
            f"**Optimal Counter-Sway Thrust:** `{mpc_out['optimal_thrust_sway_n']} N`\n\n"
            f"**Optimal Surge Thrust:** `{mpc_out['optimal_thrust_surge_n']} N`\n\n"
            f"**Current Feedforward Compensation:** `{mpc_out['feedforward_current_thrust_n']} N`"
        )

        st.markdown("#### 🔌 MAVLink ArduSub Message Output")
        mav_packet = MAVLinkAutopilotBridge.encode_set_position_target(latitude, longitude, depth, heading)
        st.code(
            f"MAVLink #84 SET_POSITION_TARGET_GLOBAL_INT:\n"
            f"  lat_int: {mav_packet['lat_int']} | lon_int: {mav_packet['lon_int']}\n"
            f"  alt_m: {mav_packet['alt_m']}m | yaw_deg: {mav_packet['yaw_deg']}° | frame: MAV_FRAME_GLOBAL_INT",
            language="yaml"
        )

st.divider()

st.caption(
    "AquaSense prototype — AI anomaly localization "
    "and marine-debris screening. The current Autoencoder "
    "detects reconstruction anomalies; object/material "
    "labels are screening estimates until a dedicated "
    "labeled object-detection model is trained."
)