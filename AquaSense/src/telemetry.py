"""
AquaSense — Ultra-Low Bandwidth Acoustic Telemetry Encoder

USP 2: Converts visual bounding box coordinates and reconstruction error
metrics into a sub-kilobyte binary/JSON telemetry vector, achieving
~99.9% bandwidth reduction for transmission over underwater acoustic modems.

Typical savings: < 0.2 KB telemetry vs 250+ KB raw image frame.
"""

import json
from datetime import datetime, timezone


def encode_acoustic_telemetry(
    mse_score,
    threshold,
    detections,
    lat=12.9716,
    lon=77.5946,
    raw_img_bytes=250000,
    raw_frame_shape=None
):
    """
    Compresses detection metadata into an ultra-low bandwidth acoustic vector.

    Extracts only the essential telemetry fields (timestamp, GPS, anomaly status,
    bounding box coordinates) and serializes into a compact JSON string. This
    payload is designed for transmission over underwater acoustic modems where
    bandwidth is severely constrained (typically 1-10 kbps).

    Args:
        mse_score (float): Frame reconstruction MSE from autoencoder.
        threshold (float): Calibrated anomaly threshold.
        detections (list): Detection list — each item can be:
            - tuple/list: (label, area, aspect_ratio, ...)
            - dict: {"type": ..., "area": ..., "aspect_ratio": ..., "bbox": ...}
        lat (float): AUV latitude coordinate.
        lon (float): AUV longitude coordinate.
        raw_img_bytes (int): Size of the raw image in bytes (for savings calc).
        raw_frame_shape (tuple): Optional shape tuple (H, W, C) to compute raw_img_bytes.

    Returns:
        dict: {
            "payload_json": str,       # Compact JSON string
            "size_kb": float,          # Payload size in kilobytes
            "size_bytes": int,         # Payload size in bytes
            "savings_pct": float,      # Bandwidth reduction percentage
            "raw_img_bytes": int       # Reference raw image size
        }
    """
    if raw_frame_shape is not None:
        try:
            prod = 1
            for dim in raw_frame_shape:
                prod *= int(dim)
            raw_img_bytes = prod
        except Exception:
            pass
    # Build compact target list from detections
    compact_targets = []
    for d in detections:
        if isinstance(d, dict):
            # Dict-style detection from AquaSenseInference
            lbl = d.get("type", d.get("label", "UNK"))
            # Truncate label for bandwidth savings
            lbl_short = lbl[:12] if len(lbl) > 12 else lbl
            conf_val = d.get("confidence_percent", round(float(d.get("confidence", 0.0)) * 100.0, 1))
            target_item = {
                "l": lbl_short,
                "b": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox else [0, 0, 0, 0],
                "a": round(float(d.get("area", 0)), 0),
                "r": round(float(d.get("aspect_ratio", 0)), 2),
                "c": round(float(conf_val), 1)
            }
            if "target_lat" in d and "target_lon" in d:
                target_item["g"] = [round(float(d["target_lat"]), 5), round(float(d["target_lon"]), 5)]
            compact_targets.append(target_item)
        elif isinstance(d, (list, tuple)):
            # Tuple-style detection: (label, area, aspect_ratio, ...)
            lbl = str(d[0])[:12] if len(str(d[0])) > 12 else str(d[0])
            compact_targets.append({
                "l": lbl,
                "a": round(float(d[1]), 0) if len(d) > 1 else 0,
                "r": round(float(d[2]), 2) if len(d) > 2 else 0
            })

    # Assemble compact payload
    compact_payload = {
        "ts": datetime.now(timezone.utc).strftime("%H%M%S"),
        "gps": [round(lat, 5), round(lon, 5)],
        "s": 1 if mse_score > threshold else 0,
        "mse": round(float(mse_score), 5),
        "th": round(float(threshold), 5),
        "n": len(compact_targets),
        "t": compact_targets
    }

    # Serialize with minimal whitespace
    json_str = json.dumps(compact_payload, separators=(',', ':'))
    telemetry_bytes = len(json_str.encode('utf-8'))

    # Calculate bandwidth savings
    bandwidth_saved = ((raw_img_bytes - telemetry_bytes) / raw_img_bytes) * 100

    return {
        "payload_json": json_str,
        "size_kb": round(telemetry_bytes / 1024.0, 4),
        "size_bytes": telemetry_bytes,
        "savings_pct": round(bandwidth_saved, 2),
        "raw_img_bytes": raw_img_bytes
    }
