"""
AquaSense — Submerged Submarine Navigation & INS/DVL Dead-Reckoning Engine

Replaces physically impossible submerged GPS with genuine subsea navigation:
1. Doppler Velocity Log (DVL): 3-axis bottom-track acoustic velocity [u, v, w]
   and seafloor altitude from 4-beam acoustic correlation.
2. Inertial Navigation System (INS): Ring Laser Gyro / FOG true heading, pitch, roll.
3. Hydrostatic Depth Gauge: Pressure-derived submarine submersion depth.
4. Direction Cosine Matrix (DCM): Rotates body velocities into North-East-Down (NED).
5. Geodetic Dead-Reckoning: Integrates displacement from last known surface fix (Datum).
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# WGS-84 Mean Earth Radius
EARTH_RADIUS_M = 6378137.0


class SubseaNavigationState:
    """Represents a validated instantaneous submarine subsea navigation state."""

    def __init__(
        self,
        northing_m: float,
        easting_m: float,
        depth_m: float,
        heading_deg: float,
        pitch_deg: float,
        roll_deg: float,
        surge_mps: float,
        sway_mps: float,
        heave_mps: float,
        altitude_m: float,
        bottom_lock: bool,
        datum_lat: float,
        datum_lon: float,
        dive_time_seconds: float,
        accumulated_drift_m: float
    ):
        self.northing_m = northing_m
        self.easting_m = easting_m
        self.depth_m = depth_m
        self.heading_deg = heading_deg
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.surge_mps = surge_mps
        self.sway_mps = sway_mps
        self.heave_mps = heave_mps
        self.altitude_m = altitude_m
        self.bottom_lock = bottom_lock
        self.datum_lat = datum_lat
        self.datum_lon = datum_lon
        self.dive_time_seconds = dive_time_seconds
        self.accumulated_drift_m = accumulated_drift_m

        # Compute true geographic lat/lon from surface datum + accumulated subsea NED displacement
        d_lat = (self.northing_m / EARTH_RADIUS_M) * (180.0 / math.pi)
        d_lon = (self.easting_m / (EARTH_RADIUS_M * math.cos(math.radians(self.datum_lat)))) * (180.0 / math.pi)
        self.latitude = self.datum_lat + d_lat
        self.longitude = self.datum_lon + d_lon

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "SUBMERGED_INS_DVL_DEAD_RECKONING",
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "depth_m": round(self.depth_m, 2),
            "northing_m": round(self.northing_m, 2),
            "easting_m": round(self.easting_m, 2),
            "heading_deg": round(self.heading_deg, 2),
            "pitch_deg": round(self.pitch_deg, 2),
            "roll_deg": round(self.roll_deg, 2),
            "surge_velocity_mps": round(self.surge_mps, 3),
            "sway_velocity_mps": round(self.sway_mps, 3),
            "heave_velocity_mps": round(self.heave_mps, 3),
            "altitude_above_seafloor_m": round(self.altitude_m, 2),
            "dvl_bottom_lock": self.bottom_lock,
            "estimated_accuracy_m": round(max(0.5, self.accumulated_drift_m), 2),
            "submerged_elapsed_s": round(self.dive_time_seconds, 1),
            "surface_datum": [self.datum_lat, self.datum_lon]
        }


class INSDVLDeadReckoningEngine:
    """
    Subsea dead-reckoning navigator replacing surface GPS.
    Integrates high-rate DVL velocity vectors and INS gyros.
    """

    def __init__(
        self,
        surface_datum_lat: float = 12.2958,
        surface_datum_lon: float = 76.6394,
        dvl_error_percent: float = 0.002  # Tactical-grade DVL: 0.2% distance traveled
    ):
        self.datum_lat = surface_datum_lat
        self.datum_lon = surface_datum_lon
        self.dvl_error_factor = dvl_error_percent

        # Accumulated subsea local Cartesian frame (NED) in meters
        self.northing_m = 0.0
        self.easting_m = 0.0
        self.depth_m = 45.0  # Initial submerged depth

        # Orientation
        self.heading_deg = 90.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

        # Velocities
        self.surge_mps = 2.0
        self.sway_mps = 0.0
        self.heave_mps = 0.0
        self.altitude_m = 12.0
        self.bottom_lock = True

        self.dive_time_s = 0.0
        self.total_distance_m = 0.0

    def set_surface_datum(self, lat: float, lon: float):
        """Sets the last known surface GNSS fix prior to submergence."""
        self.datum_lat = lat
        self.datum_lon = lon
        self.northing_m = 0.0
        self.easting_m = 0.0
        self.total_distance_m = 0.0
        self.dive_time_s = 0.0

    def update_telemetry(
        self,
        heading_deg: float,
        pitch_deg: float,
        roll_deg: float,
        surge_mps: float,
        sway_mps: float,
        heave_mps: float,
        depth_m: float,
        altitude_m: float,
        bottom_lock: bool = True,
        dt_seconds: float = 1.0
    ) -> SubseaNavigationState:
        """
        Integrates a single DVL/INS telemetry epoch into the subsea trajectory.
        Uses 3D Direction Cosine Matrix (DCM) rotation from body to NED.
        """
        self.heading_deg = heading_deg % 360.0
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.surge_mps = surge_mps
        self.sway_mps = sway_mps
        self.heave_mps = heave_mps
        self.depth_m = depth_m
        self.altitude_m = altitude_m
        self.bottom_lock = bottom_lock
        self.dive_time_s += dt_seconds

        # Body velocity vector
        v_body = np.array([surge_mps, sway_mps, heave_mps])

        # Direction Cosine Matrix Euler rotation R_b^n (ZYX convention)
        psi = math.radians(heading_deg)
        theta = math.radians(pitch_deg)
        phi = math.radians(roll_deg)

        r11 = math.cos(theta) * math.cos(psi)
        r12 = math.sin(phi) * math.sin(theta) * math.cos(psi) - math.cos(phi) * math.sin(psi)
        r13 = math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)

        r21 = math.cos(theta) * math.sin(psi)
        r22 = math.sin(phi) * math.sin(theta) * math.sin(psi) + math.cos(phi) * math.cos(psi)
        r23 = math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)

        r31 = -math.sin(theta)
        r32 = math.sin(phi) * math.cos(theta)
        r33 = math.cos(phi) * math.cos(theta)

        R_b_n = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        # Rotate velocity to NED frame
        v_ned = R_b_n @ v_body
        v_north = float(v_ned[0])
        v_east = float(v_ned[1])

        # Integrate displacements
        delta_north = v_north * dt_seconds
        delta_east = v_east * dt_seconds
        step_distance = math.sqrt(delta_north ** 2 + delta_east ** 2)

        self.northing_m += delta_north
        self.easting_m += delta_east
        self.total_distance_m += step_distance

        # Tactical DVL drift estimation: 0.2% of distance + gyro drift
        accumulated_drift = (self.total_distance_m * self.dvl_error_factor) + (0.05 * (self.dive_time_s / 3600.0))

        return SubseaNavigationState(
            northing_m=self.northing_m,
            easting_m=self.easting_m,
            depth_m=self.depth_m,
            heading_deg=self.heading_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            surge_mps=self.surge_mps,
            sway_mps=self.sway_mps,
            heave_mps=self.heave_mps,
            altitude_m=self.altitude_m,
            bottom_lock=self.bottom_lock,
            datum_lat=self.datum_lat,
            datum_lon=self.datum_lon,
            dive_time_seconds=self.dive_time_s,
            accumulated_drift_m=accumulated_drift
        )

    def parse_dvl_binary_packet(self, packet_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Parses standard Teledyne RDI / Nortek DVL binary velocity telegrams.
        Structure: [Header 2B, Velocity X 4B, Velocity Y 4B, Velocity Z 4B, Altitude 4B, Status 1B].
        """
        if len(packet_bytes) < 19:
            return None
        try:
            import struct
            header, vx, vy, vz, alt, status = struct.unpack(">HffffB", packet_bytes[:19])
            if header != 0x7F7F:  # Standard DVL sync word
                return None
            return {
                "surge_mps": vx,
                "sway_mps": vy,
                "heave_mps": vz,
                "altitude_m": alt,
                "bottom_lock": bool(status & 0x01)
            }
        except Exception:
            return None
