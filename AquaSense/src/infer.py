import json
from pathlib import Path
import cv2
import numpy as np
import torch
try:
    from src.model import ConvAutoencoder, AquaSenseAutoencoder
    from src.confidence import CalibratedConfidenceScorer
    from src.geotagging import GeotaggingEngine
    from src.features import ShadowFeatureExtractor
except ModuleNotFoundError:
    from model import ConvAutoencoder, AquaSenseAutoencoder
    from confidence import CalibratedConfidenceScorer
    from geotagging import GeotaggingEngine
    from features import ShadowFeatureExtractor


IMAGE_SIZE = 224
MIN_CONTOUR_AREA = 100

# Color specifications per Master Prompt:
# Green for Known Type, Red for Unknown Anomaly (BGR format for OpenCV)
COLOR_KNOWN = (0, 220, 50)       # Vibrant Green
COLOR_UNKNOWN = (30, 40, 240)    # Urgent Red
COLOR_CLEAR = (180, 220, 60)     # Muted Cyan-Green


class AquaSenseInference:
    """
    AquaSense Two-Stage Hybrid Underwater Object Detection & Anomaly System.

    Stage 1 — Unsupervised Anomaly Engine (Reconstruction ConvAutoencoder):
        Learns normal seabed patterns. High pixel-wise reconstruction residual
        error indicates an unknown anomaly or submerged foreign object.

    Stage 2 — Hybrid Object & Type Classification Engine:
        Isolates residual anomaly contours via Otsu thresholding, extracts
        geometric and contrast feature vectors, classifies known marine structures
        (Shipwrecks, Aircraft Components, Fauna, Pipelines) vs Unknown Anomalies,
        evaluates Calibrated Multi-Modal Confidence, projects True Seabed Geotags,
        and renders Green/Red bounding boxes with itemized telemetry.
    """

    def __init__(self, model_path=None, threshold_path=None, device=None):
        self.project_dir = Path(__file__).resolve().parent.parent

        self.model_path = (
            Path(model_path)
            if model_path
            else self.project_dir / "models" / "aquasense_ae.pth"
        )
        self.threshold_path = (
            Path(threshold_path)
            if threshold_path
            else self.project_dir / "models" / "threshold.json"
        )

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load calibrated threshold
        self.threshold, self.threshold_meta = self._load_threshold()

        # Load Stage 1 ConvAutoencoder
        self.model = ConvAutoencoder().to(self.device)
        if self.model_path.exists():
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
            except Exception as e:
                print(f"[WARN] Failed to load model weights from {self.model_path}: {e}")
                self.model_loaded = False
        else:
            print(f"[WARN] Model file not found at {self.model_path}")
            self.model_loaded = False

        self.model.eval()

        # Initialize Calibrated Confidence Scorer
        mean_val = float(self.threshold_meta.get("mean_validation_error", 0.008666))
        std_val = float(self.threshold_meta.get("std_validation_error", 0.006027))
        self.confidence_scorer = CalibratedConfidenceScorer(
            mean_clean_mse=mean_val,
            std_clean_mse=std_val,
            threshold=self.threshold
        )

        # Initialize Geotagging Engine & Shadow Extractor
        self.geotagging_engine = GeotaggingEngine()
        self.shadow_extractor = ShadowFeatureExtractor()

    def _load_threshold(self):
        """Loads the calibrated threshold from models/threshold.json."""
        default_threshold = 0.026747
        meta = {}
        if self.threshold_path.exists():
            try:
                with open(self.threshold_path, "r") as f:
                    data = json.load(f)
                    threshold = float(data.get("threshold", default_threshold))
                    return threshold, data
            except Exception as e:
                print(f"[WARN] Error reading threshold.json: {e}")
        return default_threshold, meta

    # Stage 1: Preprocessing & Anomaly Detection

    def preprocess(self, frame):
        """
        Standardizes input frame:
        - Grayscale conversion
        - Resized to 224x224
        - Normalized to [0.0, 1.0]
        - Returns PyTorch tensor [1, 1, 224, 224]
        """
        if frame is None:
            raise ValueError("Input frame is None")

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), resized

    def predict(self, frame):
        """
        Executes Stage 1 forward pass:
        Returns:
            anomaly_score (float): Frame MSE
            reconstruction (np.ndarray): Reconstructed image (224x224 uint8)
            residual (np.ndarray): Absolute pixel difference map |I - I_hat| (224x224 uint8)
        """
        tensor, resized = self.preprocess(frame)

        with torch.no_grad():
            reconstructed_tensor = self.model(tensor)
            mse_loss = torch.mean((tensor - reconstructed_tensor) ** 2)

        reconstruction = (
            reconstructed_tensor.squeeze().cpu().numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)

        residual = cv2.absdiff(resized, reconstruction)
        anomaly_score = float(mse_loss.item())

        return anomaly_score, reconstruction, residual

    # Stage 2: Hybrid Object & Type Classification Engine

    def classify_contour(self, contour, residual_map, original_gray, frame_w, frame_h, frame_mse=None):
        """
        Decision Matrix for Object Type Identification based on contour geometry:
        - Contour Area (A)
        - Bounding Box Width (w) and Height (h)
        - Aspect Ratio (w / h)
        - Contrast / Sparsity Score
        - Oriented Bounding Box (OBB)
        - Calibrated Multi-Modal Confidence Scoring
        """
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        long_dim = max(w, h)
        short_dim = min(w, h)
        elongation = float(long_dim) / float(short_dim) if short_dim > 0 else 1.0

        # Contrast / Sparsity extraction
        mask = np.zeros(residual_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_anomaly = float(cv2.mean(residual_map, mask=mask)[0])

        # Contrast of the object in the original grayscale image
        mean_obj_intensity = float(cv2.mean(original_gray, mask=mask)[0])
        # Background contrast
        inv_mask = cv2.bitwise_not(mask)
        mean_bg_intensity = float(cv2.mean(original_gray, mask=inv_mask)[0])
        contrast_score = abs(mean_obj_intensity - mean_bg_intensity) / 255.0

        # Boundary checks for spanning pipeline/cable
        spans_width = (w / float(IMAGE_SIZE)) >= 0.70
        spans_height = (h / float(IMAGE_SIZE)) >= 0.70
        touches_edges = (x <= 6 or y <= 6 or (x + w) >= IMAGE_SIZE - 6 or (y + h) >= IMAGE_SIZE - 6)

        # Decision Matrix Rules:
        if (spans_width or spans_height) and elongation >= 2.2 and touches_edges:
            object_type = "Linear Pipeline / Cable Track"
            status = "KNOWN TYPE"
        elif area > 2500 and (aspect_ratio > 2.0 or aspect_ratio < 0.5):
            object_type = "Shipwreck Frame / Linear Debris"
            status = "KNOWN TYPE"
        elif (1200 <= area <= 2500) and (elongation >= 1.6 or contrast_score > 0.20):
            object_type = "Submerged Aircraft Component"
            status = "KNOWN TYPE"
        elif area < 500 and (contrast_score > 0.18 or mean_anomaly > 40):
            object_type = "Submerged Marine Object / Fauna"
            status = "KNOWN TYPE"
        else:
            object_type = "Unclassified Submerged Threat"
            status = "UNKNOWN ANOMALY"

        # Oriented Bounding Box (OBB) - Eliminates non-axial background water noise
        rect = cv2.minAreaRect(contour)
        (obx, oby), (obw, obh), ob_angle = rect
        obb_info = {
            "center": (round(float(obx), 1), round(float(oby), 1)),
            "length": round(float(max(obw, obh)), 1),
            "breadth": round(float(min(obw, obh)), 1),
            "angle_deg": round(float(ob_angle), 1)
        }

        # Calibrated Confidence Evaluation
        current_mse = frame_mse if frame_mse is not None else self.threshold
        shadow_score = self.shadow_extractor.score(original_gray)
        spans_frame = bool(spans_width or spans_height)

        conf_eval = self.confidence_scorer.evaluate(
            frame_mse=current_mse,
            object_type=object_type,
            status=status,
            area=area,
            aspect_ratio=aspect_ratio,
            elongation=elongation,
            contrast_score=contrast_score,
            mean_anomaly=mean_anomaly,
            spans_frame=spans_frame,
            shadow_score=shadow_score
        )

        return {
            "type": object_type,
            "status": status,
            "confidence": conf_eval["composite_confidence"],
            "confidence_percent": conf_eval["confidence_percent"],
            "confidence_breakdown": conf_eval,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "elongation": elongation,
            "contrast": contrast_score,
            "mean_anomaly": mean_anomaly,
            "bbox_224": (x, y, w, h),
            "obb": obb_info
        }

    def detect_and_classify(self, frame, residual, frame_mse=None, nav_telemetry=None):
        """
        Executes Stage 2:
        1. Generates JET colormap heatmap
        2. Applies Otsu binary thresholding on residual map
        3. Filters noise contours (A < 100 px)
        4. Classifies each contour with Calibrated Confidence and Geotags
        """
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / float(IMAGE_SIZE)
        scale_y = orig_h / float(IMAGE_SIZE)

        # Convert frame to grayscale for feature analysis
        if len(frame.shape) == 3:
            orig_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            orig_gray_224 = cv2.resize(orig_gray, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            orig_gray_224 = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        # Resolve AUV coordinate state and provenance
        nav_coords = self.geotagging_engine.resolve_coordinates(auv_telemetry=nav_telemetry)
        veh_lat = nav_coords["latitude"]
        veh_lon = nav_coords["longitude"]
        veh_depth = nav_coords["depth_m"]
        heading_deg = float(nav_telemetry.get("heading", 0.0)) if nav_telemetry else 0.0
        altitude_m = float(nav_telemetry.get("altitude_m", 5.0)) if nav_telemetry else 5.0

        # JET colormap heatmap
        heatmap_224 = cv2.applyColorMap(residual, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap_224, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Otsu thresholding on residual map to isolate anomaly contours
        blurred_residual = cv2.GaussianBlur(residual, (5, 5), 0)
        otsu_val, otsu_mask = cv2.threshold(
            blurred_residual,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Noise-floor gate: If Otsu threshold is below subtle seabed background texture (< 24)
        # or peak residual is insignificant (< 30), treat frame as uniform clean seabed.
        if otsu_val < 22 or residual.max() < 28:
            otsu_mask = np.zeros_like(residual)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            cleaned_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter noise contours (A < 100 px)
        valid_contours = []
        for c in contours:
            if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
                c_mask = np.zeros(residual.shape, dtype=np.uint8)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                mean_res = float(cv2.mean(residual, mask=c_mask)[0])
                if mean_res >= 20.0:  # Must have genuine residual contrast above baseline
                    valid_contours.append(c)

        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        detections = []
        bounded_frame = frame.copy()

        for idx, contour in enumerate(valid_contours, start=1):
            profile = self.classify_contour(
                contour,
                residual,
                orig_gray_224,
                orig_w,
                orig_h,
                frame_mse=frame_mse
            )

            # Scale bounding box to original frame coordinates
            x224, y224, w224, h224 = profile["bbox_224"]
            box_x = int(round(x224 * scale_x))
            box_y = int(round(y224 * scale_y))
            box_w = int(round(w224 * scale_x))
            box_h = int(round(h224 * scale_y))

            # Clamp boundaries
            box_x = max(0, min(box_x, orig_w - 1))
            box_y = max(0, min(box_y, orig_h - 1))
            box_w = min(box_w, orig_w - box_x)
            box_h = min(box_h, orig_h - box_y)

            # Assign Green for Known Type, Red for Unknown Anomaly
            is_known = (profile["status"] == "KNOWN TYPE")
            box_color = COLOR_KNOWN if is_known else COLOR_UNKNOWN

            label_prefix = "KNOWN TYPE" if is_known else "UNKNOWN ANOMALY"
            conf_pct = profile["confidence_percent"]
            full_label = f"{label_prefix}: {profile['type']} ({conf_pct}%)"

            # Geotarget Seafloor Projection
            bbox_centroid = (box_x + box_w / 2.0, box_y + box_h / 2.0)
            target_geo = self.geotagging_engine.project_pixel_to_seafloor(
                vehicle_lat=veh_lat,
                vehicle_lon=veh_lon,
                vehicle_heading_deg=heading_deg,
                altitude_m=altitude_m,
                bbox_center_px=bbox_centroid,
                frame_shape=(orig_h, orig_w)
            )

            # Draw bounding box
            thickness = max(2, int(min(orig_w, orig_h) / 300))
            cv2.rectangle(
                bounded_frame,
                (box_x, box_y),
                (box_x + box_w, box_y + box_h),
                box_color,
                thickness
            )

            # Corner accents for tactical sonar aesthetics
            corner_len = max(10, int(min(box_w, box_h) * 0.15))
            c_thick = thickness + 2
            # Top-left
            cv2.line(bounded_frame, (box_x, box_y), (box_x + corner_len, box_y), box_color, c_thick)
            cv2.line(bounded_frame, (box_x, box_y), (box_x, box_y + corner_len), box_color, c_thick)
            # Top-right
            cv2.line(bounded_frame, (box_x + box_w, box_y), (box_x + box_w - corner_len, box_y), box_color, c_thick)
            cv2.line(bounded_frame, (box_x + box_w, box_y), (box_x + box_w, box_y + corner_len), box_color, c_thick)
            # Bottom-left
            cv2.line(bounded_frame, (box_x, box_y + box_h), (box_x + corner_len, box_y + box_h), box_color, c_thick)
            cv2.line(bounded_frame, (box_x, box_y + box_h), (box_x, box_y + box_h - corner_len), box_color, c_thick)
            # Bottom-right
            cv2.line(bounded_frame, (box_x + box_w, box_y + box_h), (box_x + box_w - corner_len, box_y + box_h), box_color, c_thick)
            cv2.line(bounded_frame, (box_x + box_w, box_y + box_h), (box_x + box_w, box_y + box_h - corner_len), box_color, c_thick)

            # Render label tag above box
            font_scale = max(0.45, min(orig_w, orig_h) / 1100.0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(full_label, font, font_scale, 1)

            tag_y1 = max(0, box_y - th - 8)
            tag_y2 = box_y
            tag_x1 = box_x
            tag_x2 = min(orig_w, box_x + tw + 12)

            cv2.rectangle(bounded_frame, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
            cv2.putText(
                bounded_frame,
                full_label,
                (tag_x1 + 6, tag_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            detection_entry = {
                "index": idx,
                "type": profile["type"],
                "status": profile["status"],
                "confidence": profile["confidence"],
                "confidence_percent": profile["confidence_percent"],
                "confidence_breakdown": profile["confidence_breakdown"],
                "uncertainty_index": profile["confidence_breakdown"]["uncertainty_index"],
                "area": profile["area"],
                "aspect_ratio": profile["aspect_ratio"],
                "contrast": profile["contrast"],
                "bbox": (box_x, box_y, box_w, box_h),
                "obb": profile["obb"],
                "is_known": is_known,
                "color_bgr": box_color,
                "target_lat": target_geo["target_lat"],
                "target_lon": target_geo["target_lon"],
                "depth_m": veh_depth,
                "geotag_source": nav_coords["source"],
                "accuracy_m": nav_coords["accuracy_m"],
                "offset_north_m": target_geo["offset_north_m"],
                "offset_east_m": target_geo["offset_east_m"]
            }
            detections.append(detection_entry)

        # Overlay residual heatmap on original frame
        alpha = 0.40
        overlay = cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "bounded_frame": bounded_frame,
            "mask": cleaned_mask,
            "detections": detections,
            "total_detections": len(detections),
            "nav_coords": nav_coords
        }

    # Complete Two-Stage Pipeline

    def analyze(self, frame, nav_telemetry=None):
        """
        Executes end-to-end AquaSense Dual-Stage Pipeline:
        1. Stage 1: ConvAutoencoder Anomaly Detection & MSE scoring vs Calibrated Threshold.
        2. Stage 2: Otsu isolation, geometric profiling, Calibrated Confidence, Geotag projection.
        """
        anomaly_score, reconstruction, residual = self.predict(frame)
        stage2_results = self.detect_and_classify(
            frame, residual, frame_mse=anomaly_score, nav_telemetry=nav_telemetry
        )

        if anomaly_score > self.threshold:
            is_anomaly = True
            global_status = "ANOMALY DETECTED"
        else:
            significant_detections = [
                d for d in stage2_results["detections"]
                if d["is_known"] or d["area"] >= 800 or d["contrast"] >= 0.22
            ]
            if len(significant_detections) > 0:
                is_anomaly = True
                global_status = "ANOMALY DETECTED"
                stage2_results["detections"] = significant_detections
                stage2_results["total_detections"] = len(significant_detections)
            else:
                is_anomaly = False
                global_status = "SEABED CLEAR"
                stage2_results["detections"] = []
                stage2_results["total_detections"] = 0
                stage2_results["bounded_frame"] = frame.copy()

        # Fallback if anomaly score is above threshold but no single contour exceeded MIN_CONTOUR_AREA
        if is_anomaly and stage2_results["total_detections"] == 0:
            orig_h, orig_w = frame.shape[:2]
            nav_coords = stage2_results.get("nav_coords", self.geotagging_engine.resolve_coordinates())
            stage2_results["detections"].append({
                "index": 1,
                "type": "Diffuse Unclassified Submerged Threat",
                "status": "UNKNOWN ANOMALY",
                "confidence": 0.80,
                "confidence_percent": 80.0,
                "confidence_breakdown": {
                    "composite_confidence": 0.80,
                    "uncertainty_index": 0.25,
                    "quality_tier": "HIGH"
                },
                "uncertainty_index": 0.25,
                "area": float(IMAGE_SIZE * IMAGE_SIZE * 0.1),
                "aspect_ratio": 1.0,
                "contrast": 0.20,
                "bbox": (0, 0, orig_w, orig_h),
                "obb": {"center": (orig_w/2, orig_h/2), "length": orig_w, "breadth": orig_h, "angle_deg": 0.0},
                "is_known": False,
                "color_bgr": COLOR_UNKNOWN,
                "target_lat": nav_coords["latitude"],
                "target_lon": nav_coords["longitude"],
                "depth_m": nav_coords["depth_m"],
                "geotag_source": nav_coords["source"],
                "accuracy_m": nav_coords["accuracy_m"],
                "offset_north_m": 0.0,
                "offset_east_m": 0.0
            })
            stage2_results["total_detections"] = 1

        return {
            "anomaly_score": anomaly_score,
            "threshold": self.threshold,
            "is_anomaly": is_anomaly,
            "global_status": global_status,
            "reconstruction": reconstruction,
            "residual": residual,
            "heatmap": stage2_results["heatmap"],
            "overlay": stage2_results["overlay"],
            "bounded_frame": stage2_results["bounded_frame"],
            "detections": stage2_results["detections"],
            "total_detections": stage2_results["total_detections"],
            "mask": stage2_results["mask"],
            "nav_coords": stage2_results.get("nav_coords")
        }