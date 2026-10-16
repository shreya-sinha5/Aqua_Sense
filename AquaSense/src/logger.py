import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
IMG_DIR = LOG_DIR / "saved_images"
CSV_PATH = LOG_DIR / "detection_history.csv"

CSV_HEADERS = [
    "Timestamp",
    "Image_Filename",
    "MSE_Score",
    "Threshold",
    "Status",
    "Total_Anomalies",
    "Identified_Types",
    "Latitude",
    "Longitude",
    "Depth_m",
    "Geotag_Source",
    "Confidence_Pct"
]


def init_logger():
    """Initializes the logging directory structure and CSV audit file with seamless migration."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    else:
        # Check and migrate legacy CSV if older 7-column schema is detected
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows and len(rows[0]) < len(CSV_HEADERS):
                upgraded_rows = [CSV_HEADERS]
                for r in rows[1:]:
                    if len(r) < len(CSV_HEADERS):
                        r = r + ["N/A"] * (len(CSV_HEADERS) - len(r))
                    upgraded_rows.append(r[:len(CSV_HEADERS)])
                with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(upgraded_rows)
        except Exception:
            pass


def _ensure_bgr(img):
    """Converts RGB/Grayscale image array to BGR format for cv2.imwrite."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] == 3:
            # Assume RGB input as specified in interface
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def _format_detection_types(detections):
    """Formats detection entries into a clean, delimited audit string."""
    if not detections:
        return "None"

    formatted = []
    for d in detections:
        if isinstance(d, dict):
            status = d.get("status", "ANOMALY")
            dtype = d.get("type", "Unknown")
            formatted.append(f"{status}: {dtype}")
        elif isinstance(d, (list, tuple)):
            formatted.append(str(d[0]))
        else:
            formatted.append(str(d))

    return "; ".join(formatted) if formatted else "None"


def save_detection_event(orig_rgb, heat_rgb, det_rgb, mse_score, threshold, detections):
    """
    Saves visual artifacts (original, heatmap, bounded detection) to disk
    and records structured forensic metadata in detection_history.csv.

    Args:
        orig_rgb (np.ndarray): Original input image (RGB)
        heat_rgb (np.ndarray): Stage 1 JET error heatmap (RGB)
        det_rgb (np.ndarray): Stage 2 bounded detection frame (RGB)
        mse_score (float): Frame reconstruction MSE
        threshold (float): Calibrated anomaly threshold
        detections (list): Detected object list (dicts, tuples, or strings)

    Returns:
        tuple: (CSV_PATH, det_filename)
    """
    init_logger()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Ensure unique filename even if multiple calls happen in the same second
    det_filename = f"det_{timestamp}.jpg"
    orig_filename = f"orig_{timestamp}.jpg"
    heat_filename = f"heat_{timestamp}.jpg"

    target_det_path = IMG_DIR / det_filename
    if target_det_path.exists():
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        det_filename = f"det_{timestamp}.jpg"
        orig_filename = f"orig_{timestamp}.jpg"
        heat_filename = f"heat_{timestamp}.jpg"

    # Save visual artifacts (converting RGB to BGR for OpenCV write)
    if orig_rgb is not None:
        cv2.imwrite(str(IMG_DIR / orig_filename), _ensure_bgr(orig_rgb))
    if heat_rgb is not None:
        cv2.imwrite(str(IMG_DIR / heat_filename), _ensure_bgr(heat_rgb))
    if det_rgb is not None:
        cv2.imwrite(str(IMG_DIR / det_filename), _ensure_bgr(det_rgb))

    # Anomaly status calculation
    status = "ANOMALY DETECTED" if mse_score > threshold or (detections and len(detections) > 0) else "SEABED CLEAR"
    types_str = _format_detection_types(detections)
    total_anomalies = len(detections) if detections else 0

    # Extract primary detection metadata for geotagging and confidence
    primary_lat = 0.0
    primary_lon = 0.0
    primary_depth = 0.0
    geotag_source = "N/A"
    primary_conf_pct = 0.0

    if detections and len(detections) > 0 and isinstance(detections[0], dict):
        first_det = detections[0]
        primary_lat = float(first_det.get("target_lat", first_det.get("latitude", 0.0)))
        primary_lon = float(first_det.get("target_lon", first_det.get("longitude", 0.0)))
        primary_depth = float(first_det.get("depth_m", 0.0))
        geotag_source = str(first_det.get("geotag_source", "AUV_TELEMETRY"))
        primary_conf_pct = float(first_det.get("confidence_percent", first_det.get("confidence", 0.0) * 100.0))

    # Append structured record to CSV
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            det_filename,
            f"{mse_score:.6f}",
            f"{threshold:.6f}",
            status,
            total_anomalies,
            types_str,
            f"{primary_lat:.6f}" if primary_lat != 0.0 else "N/A",
            f"{primary_lon:.6f}" if primary_lon != 0.0 else "N/A",
            f"{primary_depth:.1f}" if primary_depth != 0.0 else "N/A",
            geotag_source,
            f"{primary_conf_pct:.1f}%" if primary_conf_pct > 0 else "N/A"
        ])

    return CSV_PATH, det_filename


def get_recent_logs(limit=20):
    """Loads and returns the latest detection history records as a DataFrame."""
    init_logger()
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=CSV_HEADERS)

    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return df
        return df.tail(limit).sort_values("Timestamp", ascending=False)
    except Exception as e:
        print(f"[WARN] Failed to load detection history: {e}")
        return pd.DataFrame(columns=CSV_HEADERS)


def get_saved_image_paths(timestamp_or_filename):
    """
    Returns existing paths for original, heatmap, and detection artifacts
    for a given timestamp or detection filename.
    """
    init_logger()
    # Strip prefixes/extensions to obtain pure timestamp
    base = str(timestamp_or_filename).replace("det_", "").replace("orig_", "").replace("heat_", "").replace(".jpg", "")
    
    orig_path = IMG_DIR / f"orig_{base}.jpg"
    heat_path = IMG_DIR / f"heat_{base}.jpg"
    det_path = IMG_DIR / f"det_{base}.jpg"

    return {
        "original": orig_path if orig_path.exists() else None,
        "heatmap": heat_path if heat_path.exists() else None,
        "detection": det_path if det_path.exists() else None,
    }
