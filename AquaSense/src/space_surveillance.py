"""
AquaSense — Surface & Space Surveillance & Multi-Modal AIS Fusion Engine

Solves civilian maritime domain awareness (Frontiers 2026 Review):
1. Satellite SAR Speckle Reduction: Lee filter & local variance smoothing.
2. Rotated Bounding Box (OBB) Representation: Eliminates water background noise.
3. AIS Telemetry Stream Parser & Temporal Interpolator (MMSI, SOG, COG).
4. Cross-Modal Anomaly & Threat Classifier:
   - "Dark Vessel" Detection (Physical radar/SAR signature with disabled AIS)
   - "AIS Spoofing / Ghost Target" (Reported transponder in empty water)
   - Route Deviation & Navigation Corridor Compliance.
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class SARProcessor:
    """
    Synthetic Aperture Radar (SAR) speckle suppression and rotated target analysis.
    """

    @staticmethod
    def lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Lee Speckle Filter for SAR images.
        Computes local mean and local variance to adaptively filter multiplicative speckle noise
        while preserving sharp ship target edges and wake boundaries.
        """
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            img_gray = image.astype(np.float32)

        # Local mean
        mean = cv2.blur(img_gray, (window_size, window_size))
        # Local squared mean
        sq_mean = cv2.blur(img_gray ** 2, (window_size, window_size))
        # Local variance
        variance = np.maximum(0.0, sq_mean - mean ** 2)

        # Overall image noise variance estimate
        overall_variance = float(np.var(img_gray))
        if overall_variance == 0:
            return np.clip(img_gray, 0, 255).astype(np.uint8)

        # Weighting factor k
        weights = variance / (variance + overall_variance + 1e-6)
        filtered = mean + weights * (img_gray - mean)

        return np.clip(filtered, 0, 255).astype(np.uint8)

    @staticmethod
    def extract_oriented_bounding_boxes(
        binary_mask: np.ndarray,
        min_area: float = 80.0
    ) -> List[Dict[str, Any]]:
        """
        Extracts Oriented Bounding Boxes (OBB) with rotation angle theta.
        Eliminates non-axial sea clutter in ship detections.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obb_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(c)  # ((cx, cy), (w, h), angle)
            (cx, cy), (w, h), angle = rect

            # Standardize length (long) and breadth (short)
            length = max(w, h)
            breadth = min(w, h)
            aspect_ratio = length / max(1.0, breadth)

            box_pts = cv2.boxPoints(rect)
            box_pts = np.int32(box_pts).tolist()

            obb_boxes.append({
                "center": (round(float(cx), 1), round(float(cy), 1)),
                "length_px": round(float(length), 1),
                "breadth_px": round(float(breadth), 1),
                "orientation_deg": round(float(angle), 1),
                "aspect_ratio": round(float(aspect_ratio), 2),
                "area_px": round(float(area), 1),
                "box_points": box_pts
            })

        return obb_boxes


class AISRecord:
    """Represents a validated Automatic Identification System broadcast record."""

    def __init__(
        self,
        mmsi: int,
        vessel_name: str,
        vessel_type: str,
        lat: float,
        lon: float,
        sog_knots: float,
        cog_deg: float,
        length_m: float = 50.0,
        timestamp: Optional[str] = None
    ):
        self.mmsi = mmsi
        self.vessel_name = vessel_name
        self.vessel_type = vessel_type
        self.lat = lat
        self.lon = lon
        self.sog_knots = sog_knots
        self.cog_deg = cog_deg
        self.length_m = length_m
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi,
            "vessel_name": self.vessel_name,
            "vessel_type": self.vessel_type,
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "sog_knots": round(self.sog_knots, 1),
            "cog_deg": round(self.cog_deg, 1),
            "length_m": round(self.length_m, 1),
            "timestamp": self.timestamp
        }


class MultiModalMaritimeFusionEngine:
    """
    Cross-references remote sensing (optical/SAR) detections against AIS broadcasts
    to identify Dark Vessels, AIS Spoofing, and navigation risks.
    """

    @staticmethod
    def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes great circle distance between two points in meters."""
        r = 6378137.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def correlate_detections_with_ais(
        self,
        optical_or_sar_detections: List[Dict[str, Any]],
        ais_vessels: List[AISRecord],
        match_radius_m: float = 300.0
    ) -> Dict[str, Any]:
        """
        Executes bipartite matching between sensor detections and reported AIS records.
        Categorizes:
        - COOPERATIVE_VESSEL: Sensor detection matched with AIS broadcast.
        - DARK_VESSEL: Physical sensor target with NO nearby AIS transponder.
        - GHOST_TARGET_OR_SPOOF: AIS broadcast with NO physical sensor confirmation.
        """
        n_det = len(optical_or_sar_detections)
        n_ais = len(ais_vessels)

        if n_det == 0 and n_ais == 0:
            return {"cooperative_vessels": [], "dark_vessels": [], "spoofed_targets": []}

        cost_matrix = np.full((n_det, n_ais), 1e6)

        for i, det in enumerate(optical_or_sar_detections):
            d_lat = det.get("target_lat", det.get("latitude", 0.0))
            d_lon = det.get("target_lon", det.get("longitude", 0.0))
            for j, ais in enumerate(ais_vessels):
                dist = self.haversine_distance_m(d_lat, d_lon, ais.lat, ais.lon)
                if dist <= match_radius_m:
                    cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_detections = set()
        matched_ais = set()
        cooperative = []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= match_radius_m:
                det = optical_or_sar_detections[r]
                ais = ais_vessels[c]
                matched_detections.add(r)
                matched_ais.add(c)
                cooperative.append({
                    "detection_id": r,
                    "mmsi": ais.mmsi,
                    "vessel_name": ais.vessel_name,
                    "vessel_type": ais.vessel_type,
                    "spatial_discrepancy_m": round(float(cost_matrix[r, c]), 1),
                    "target_lat": det.get("target_lat", det.get("latitude")),
                    "target_lon": det.get("target_lon", det.get("longitude")),
                    "status": "VERIFIED_COOPERATIVE"
                })

        # Unmatched sensor detections are dark vessels
        dark_vessels = []
        for i, det in enumerate(optical_or_sar_detections):
            if i not in matched_detections:
                dark_vessels.append({
                    "detection_id": i,
                    "target_lat": det.get("target_lat", det.get("latitude")),
                    "target_lon": det.get("target_lon", det.get("longitude")),
                    "estimated_length_px": det.get("length_px", det.get("area", 0.0)),
                    "threat_level": "CRITICAL_DARK_VESSEL",
                    "reason": "Physical radar/optical signature observed with disabled or missing AIS transponder"
                })

        # Unmatched AIS broadcasts are ghost targets / spoofing
        spoofed = []
        for j, ais in enumerate(ais_vessels):
            if j not in matched_ais:
                spoofed.append({
                    "mmsi": ais.mmsi,
                    "vessel_name": ais.vessel_name,
                    "reported_lat": ais.lat,
                    "reported_lon": ais.lon,
                    "reported_sog": ais.sog_knots,
                    "threat_level": "WARNING_AIS_SPOOFING",
                    "reason": "Active AIS transponder reported coordinates with zero physical target confirmation"
                })

        return {
            "total_sensor_detections": n_det,
            "total_ais_broadcasts": n_ais,
            "cooperative_count": len(cooperative),
            "dark_vessel_count": len(dark_vessels),
            "spoofed_target_count": len(spoofed),
            "cooperative_vessels": cooperative,
            "dark_vessels": dark_vessels,
            "spoofed_targets": spoofed
        }
