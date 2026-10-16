import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import (
    init_logger,
    save_detection_event,
    get_recent_logs,
    get_saved_image_paths,
    CSV_PATH,
    IMG_DIR
)


def test_logging_system():
    print("=" * 60)
    print("Testing AquaSense Persistent Event Logger")
    print("=" * 60)

    # 1. Initialize
    init_logger()
    assert IMG_DIR.exists(), f"IMG_DIR does not exist: {IMG_DIR}"
    assert CSV_PATH.exists(), f"CSV_PATH does not exist: {CSV_PATH}"
    print(f"[PASS] Logger initialized. CSV: {CSV_PATH}")

    # 2. Test event saving with synthetic images
    h, w = 224, 224
    dummy_orig = np.full((h, w, 3), 100, dtype=np.uint8)
    dummy_heat = np.full((h, w, 3), 150, dtype=np.uint8)
    dummy_det = np.full((h, w, 3), 200, dtype=np.uint8)

    sample_detections = [
        {"type": "Shipwreck Frame / Linear Debris", "status": "KNOWN TYPE", "confidence": 0.95, "area": 3200},
        {"type": "Unclassified Submerged Threat", "status": "UNKNOWN ANOMALY", "confidence": 0.75, "area": 1200}
    ]

    csv_path, saved_det_filename = save_detection_event(
        orig_rgb=dummy_orig,
        heat_rgb=dummy_heat,
        det_rgb=dummy_det,
        mse_score=0.031500,
        threshold=0.026747,
        detections=sample_detections
    )

    print(f"[PASS] Event saved. Filename: {saved_det_filename}")

    # Verify physical file existence
    det_path = IMG_DIR / saved_det_filename
    assert det_path.exists(), f"Detection image not found: {det_path}"
    print(f"[PASS] Detection image artifact confirmed: {det_path.name} ({det_path.stat().st_size} bytes)")

    paths = get_saved_image_paths(saved_det_filename)
    assert paths["original"] and paths["original"].exists(), "Original artifact missing"
    assert paths["heatmap"] and paths["heatmap"].exists(), "Heatmap artifact missing"
    assert paths["detection"] and paths["detection"].exists(), "Detection artifact missing"
    print("[PASS] All 3 visual artifacts verified on disk (orig, heat, det)")

    # Verify CSV metadata
    recent_df = get_recent_logs(limit=5)
    assert not recent_df.empty, "Recent logs returned empty DataFrame"
    latest_row = recent_df.iloc[0]
    print(f"[PASS] CSV row parsed: Timestamp={latest_row['Timestamp']}, Status={latest_row['Status']}, Types={latest_row['Identified_Types']}")
    assert "Shipwreck" in latest_row["Identified_Types"]

    print("=" * 60)
    print("All Persistent Logging Tests Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_logging_system()
