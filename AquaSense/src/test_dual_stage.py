import sys
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import AquaSenseInference


def main():
    print("=" * 60)
    print("AquaSense Dual-Stage Engine Verification Test")
    print("=" * 60)

    engine = AquaSenseInference()

    print(f"Device               : {engine.device}")
    print(f"Model Loaded         : {engine.model_loaded}")
    print(f"Calibrated Threshold : {engine.threshold:.6f}")
    print(f"Threshold Method     : {engine.threshold_meta.get('method', 'N/A')}")
    print()

    debris_dir = PROJECT_ROOT / "data" / "raw" / "debris"
    clean_dir = PROJECT_ROOT / "data" / "raw" / "clean"

    sample_debris = list(debris_dir.glob("*.jpg"))[:5]
    sample_clean = list(clean_dir.glob("*.jpg"))[:3]

    print(f"Testing {len(sample_debris)} debris images and {len(sample_clean)} clean images...\n")

    print("--- 1. DEBRIS SAMPLES (Expected Anomalies & Classified Objects) ---")
    for img_path in sample_debris:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = engine.analyze(frame)
        print(f"File: {img_path.name}")
        print(f"  MSE: {res['anomaly_score']:.6f} | Threshold: {res['threshold']:.6f}")
        print(f"  Status: {res['global_status']} (is_anomaly={res['is_anomaly']})")
        print(f"  Detections Count: {res['total_detections']}")
        for d in res["detections"]:
            print(f"    - [{d['status']}] {d['type']} (Conf: {d['confidence']*100:.1f}%, Area: {d['area']:.0f}px, Aspect: {d['aspect_ratio']:.2f})")
        print()

    print("--- 2. CLEAN SAMPLES (Expected Seabed Clear Baseline) ---")
    for img_path in sample_clean:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = engine.analyze(frame)
        print(f"File: {img_path.name}")
        print(f"  MSE: {res['anomaly_score']:.6f} | Threshold: {res['threshold']:.6f}")
        print(f"  Status: {res['global_status']} (is_anomaly={res['is_anomaly']})")
        print(f"  Detections Count: {res['total_detections']}")
        print()

    print("=" * 60)
    print("Dual-Stage Engine Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
