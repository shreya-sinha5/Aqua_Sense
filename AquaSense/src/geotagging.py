"""
AquaSense — Multi-Source Geotagging & Anomaly Reporting Engine

Bridges the geotagging provenance gap by ingesting spatial coordinates from:
1. Optical/Sonar image EXIF GPS tags.
2. NMEA 0183 navigation sentences ($GPGGA, $GPRMC, $GPGLL).
3. USBL (Ultra-Short Baseline) acoustic relative transponder telemetry.
4. Real-time AUV dead-reckoning state (GNSS + IMU heading + depth + altitude).
5. Ground-plane target georeferencing: projects bounding box pixel centroids
   onto true seafloor WGS-84 coordinates using sensor Field-of-View (FOV).
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ExifTags


# Earth parameters (WGS-84 ellipsoid approximation)
EARTH_RADIUS_M = 6378137.0


class GeotaggingEngine:
    """
    Manages coordinate resolution, sensor projection, and provenance metadata
    for underwater anomaly reporting.
    """

    def __init__(
        self,
        default_lat: float = 12.9716,
        default_lon: float = 77.5946,
        default_depth: float = 15.0
    ):
        self.default_lat = default_lat
        self.default_lon = default_lon
        self.default_depth = default_depth

    # 1. EXIF GPS Ingestion
    @staticmethod
    def extract_from_exif(image_path_or_bytes) -> Optional[Dict[str, Any]]:
        """
        Extracts GPS metadata from image EXIF tags if present.
        """
        try:
            if isinstance(image_path_or_bytes, (str, Path)):
                img = Image.open(image_path_or_bytes)
            else:
                img = Image.open(image_path_or_bytes)

            exif = img._getexif()
            if not exif:
                return None

            gps_info = {}
            for tag, val in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in val:
                        sub_decoded = ExifTags.GPSTAGS.get(t, t)
                        gps_info[sub_decoded] = val[t]

            if not gps_info or "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
                return None

            def _convert_to_degrees(value):
                d = float(value[0])
                m = float(value[1])
                s = float(value[2])
                return d + (m / 60.0) + (s / 3600.0)

            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat

            lon = _convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon

            altitude = float(gps_info.get("GPSAltitude", 0.0))

            return {
                "latitude": round(lat, 7),
                "longitude": round(lon, 7),
                "depth_m": max(0.0, -altitude),
                "source": "IMAGE_EXIF_GPS",
                "accuracy_m": 3.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception:
            return None

    # 2. NMEA 0183 Ingestion ($GPGGA, $GPRMC, $GPGLL)
    @staticmethod
    def parse_nmea_sentence(sentence: str) -> Optional[Dict[str, Any]]:
        """
        Parses raw NMEA 0183 GNSS string into decimal degrees.
        """
        sentence = sentence.strip()
        parts = sentence.split(",")
        if len(parts) < 6:
            return None

        talker = parts[0].upper()

        def _nmea_to_deg(raw: str, direction: str) -> float:
            if not raw or not direction:
                return 0.0
            idx = 2 if direction.upper() in ["N", "S"] else 3
            degrees = float(raw[:idx])
            minutes = float(raw[idx:])
            decimal = degrees + (minutes / 60.0)
            if direction.upper() in ["S", "W"]:
                decimal = -decimal
            return decimal

        try:
            if "$GPGGA" in talker or "$GNGGA" in talker:
                # Format: $GPGGA,time,lat,N/S,lon,E/W,fix_quality,satellites,hdop,alt,...
                lat = _nmea_to_deg(parts[2], parts[3])
                lon = _nmea_to_deg(parts[4], parts[5])
                fix_quality = int(parts[6]) if parts[6] else 0
                alt = float(parts[9]) if len(parts) > 9 and parts[9] else 0.0
                return {
                    "latitude": round(lat, 7),
                    "longitude": round(lon, 7),
                    "depth_m": max(0.0, -alt),
                    "source": "NMEA_0183_GGA",
                    "fix_quality": fix_quality,
                    "accuracy_m": 1.5 if fix_quality == 4 else (3.0 if fix_quality == 1 else 10.0),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            elif "$GPRMC" in talker or "$GNRMC" in talker:
                lat = _nmea_to_deg(parts[3], parts[4])
                lon = _nmea_to_deg(parts[5], parts[6])
                sog_knots = float(parts[7]) if parts[7] else 0.0
                cog_deg = float(parts[8]) if parts[8] else 0.0
                return {
                    "latitude": round(lat, 7),
                    "longitude": round(lon, 7),
                    "sog_knots": sog_knots,
                    "cog_deg": cog_deg,
                    "depth_m": 0.0,
                    "source": "NMEA_0183_RMC",
                    "accuracy_m": 3.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        except Exception:
            return None
        return None

    # 3. USBL Acoustic Relative Positioning
    @staticmethod
    def calculate_usbl_position(
        mothership_lat: float,
        mothership_lon: float,
        slant_range_m: float,
        azimuth_deg: float,
        elevation_deg: float
    ) -> Dict[str, Any]:
        """
        Computes subsea transponder coordinate from mothership USBL head:
        slant range, azimuth (relative to True North), and elevation.
        """
        # Convert angles to radians
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)

        # Planar horizontal distance and depth
        horizontal_dist = slant_range_m * math.cos(el_rad)
        subsea_depth = slant_range_m * math.sin(el_rad)

        # Local North and East displacements
        delta_north = horizontal_dist * math.cos(az_rad)
        delta_east = horizontal_dist * math.sin(az_rad)

        # Spherical displacement to lat/lon
        delta_lat = (delta_north / EARTH_RADIUS_M) * (180.0 / math.pi)
        delta_lon = (delta_east / (EARTH_RADIUS_M * math.cos(math.radians(mothership_lat)))) * (180.0 / math.pi)

        target_lat = mothership_lat + delta_lat
        target_lon = mothership_lon + delta_lon

        return {
            "latitude": round(target_lat, 7),
            "longitude": round(target_lon, 7),
            "depth_m": round(abs(subsea_depth), 2),
            "source": "USBL_ACOUSTIC_TELEMETRY",
            "slant_range_m": round(slant_range_m, 2),
            "azimuth_deg": round(azimuth_deg, 2),
            "accuracy_m": round(max(2.0, slant_range_m * 0.01), 2),  # 1% slant range acoustic accuracy
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # 4. Pixel-to-Geographic Seabed Projection
    @staticmethod
    def project_pixel_to_seafloor(
        vehicle_lat: float,
        vehicle_lon: float,
        vehicle_heading_deg: float,
        altitude_m: float,
        bbox_center_px: Tuple[float, float],
        frame_shape: Tuple[int, int] = (224, 224),
        horizontal_fov_deg: float = 60.0,
        vertical_fov_deg: float = 45.0
    ) -> Dict[str, Any]:
        """
        Projects a 2D bounding box center in an image frame to georeferenced
        Earth coordinates based on AUV altitude, heading, and sensor FOV.
        """
        cx_px, cy_px = bbox_center_px
        img_h, img_w = frame_shape

        # Normalized coordinates centered at optical axis [-0.5, 0.5]
        norm_x = (cx_px - (img_w / 2.0)) / img_w
        norm_y = ((img_h / 2.0) - cy_px) / img_h  # Positive is forward

        # Ground swath footprint at given altitude
        swath_w = 2.0 * altitude_m * math.tan(math.radians(horizontal_fov_deg / 2.0))
        swath_h = 2.0 * altitude_m * math.tan(math.radians(vertical_fov_deg / 2.0))

        # Local body-fixed offsets (x_body: forward, y_body: starboard)
        x_body = norm_y * swath_h
        y_body = norm_x * swath_w

        # Rotate body-fixed offsets by vehicle heading
        heading_rad = math.radians(vehicle_heading_deg)
        delta_north = x_body * math.cos(heading_rad) - y_body * math.sin(heading_rad)
        delta_east = x_body * math.sin(heading_rad) + y_body * math.cos(heading_rad)

        # Displace vehicle lat/lon to target lat/lon
        delta_lat = (delta_north / EARTH_RADIUS_M) * (180.0 / math.pi)
        delta_lon = (delta_east / (EARTH_RADIUS_M * math.cos(math.radians(vehicle_lat)))) * (180.0 / math.pi)

        target_lat = vehicle_lat + delta_lat
        target_lon = vehicle_lon + delta_lon
        radial_distance = math.sqrt(delta_north ** 2 + delta_east ** 2)

        return {
            "target_lat": round(target_lat, 7),
            "target_lon": round(target_lon, 7),
            "offset_north_m": round(delta_north, 2),
            "offset_east_m": round(delta_east, 2),
            "radial_offset_m": round(radial_distance, 2),
            "georeference_method": "SEABED_FOV_RAY_PROJECTION"
        }

    # 5. Resolve Coordinates with Provenance Fallback
    def resolve_coordinates(
        self,
        exif_source: Optional[Dict[str, Any]] = None,
        nmea_sentence: Optional[str] = None,
        usbl_data: Optional[Dict[str, Any]] = None,
        auv_telemetry: Optional[Dict[str, Any]] = None,
        manual_override: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Resolves the highest-fidelity available coordinate source according to
        strict operational precedence:
        1. USBL Acoustic Real-Time Slant Range (Subsea Highest Accuracy)
        2. In-Stream NMEA Navigation Feed
        3. Image EXIF GPS Tags
        4. AUV Dead-Reckoning Navigation State
        5. Manual / Station Coordinates (Fallback)
        """
        if usbl_data and "mothership_lat" in usbl_data:
            return self.calculate_usbl_position(
                usbl_data["mothership_lat"],
                usbl_data["mothership_lon"],
                usbl_data["slant_range_m"],
                usbl_data.get("azimuth_deg", 0.0),
                usbl_data.get("elevation_deg", 45.0)
            )

        if nmea_sentence:
            nmea_res = self.parse_nmea_sentence(nmea_sentence)
            if nmea_res:
                return nmea_res

        if exif_source and "latitude" in exif_source:
            return exif_source

        if auv_telemetry and "lat" in auv_telemetry and "lon" in auv_telemetry:
            return {
                "latitude": round(float(auv_telemetry["lat"]), 7),
                "longitude": round(float(auv_telemetry["lon"]), 7),
                "depth_m": round(float(auv_telemetry.get("depth", self.default_depth)), 2),
                "source": "AUV_DEAD_RECKONING_TELEMETRY",
                "accuracy_m": 5.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        if manual_override:
            lat, lon, depth = manual_override
            return {
                "latitude": round(lat, 7),
                "longitude": round(lon, 7),
                "depth_m": round(depth, 2),
                "source": "MANUAL_STATION_OVERRIDE",
                "accuracy_m": 15.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        return {
            "latitude": self.default_lat,
            "longitude": self.default_lon,
            "depth_m": self.default_depth,
            "source": "STATION_BENCHMARK_DEFAULT",
            "accuracy_m": 25.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # 6. Standardized Forensic Report Generator (GeoJSON)
    @staticmethod
    def generate_geojson_report(detection_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converts detection events into standardized OGC GeoJSON FeatureCollection.
        """
        features = []
        for det in detection_events:
            lat = det.get("target_lat", det.get("latitude", 0.0))
            lon = det.get("target_lon", det.get("longitude", 0.0))

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat, -abs(det.get("depth_m", 0.0))]
                },
                "properties": {
                    "label": det.get("type", "Unknown Anomaly"),
                    "status": det.get("status", "ANOMALY"),
                    "confidence_pct": det.get("confidence_percent", 0.0),
                    "mse_score": det.get("anomaly_score", 0.0),
                    "provenance": det.get("source", "UNKNOWN"),
                    "accuracy_m": det.get("accuracy_m", 0.0),
                    "timestamp": det.get("timestamp", datetime.now(timezone.utc).isoformat())
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            }
        }
