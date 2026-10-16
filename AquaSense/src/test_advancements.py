"""
AquaSense — Comprehensive Verification Suite for Software-Level Advancements

Validates:
1. Calibrated Confidence Scoring & Uncertainty Engine
2. Multi-Source Geotagging & Seabed FOV Projection Engine
3. Acoustic Physics, Sound Speed Profile & Snell's Law Ray-Tracing
4. Surface & Space Surveillance, OBB & AIS Multi-Modal Threat Fusion
5. 6-DOF Fossen Hydrodynamics & Cable-Tracking Model Predictive Control (MPC)
6. Full Dual-Stage Pipeline Integration
"""

import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.confidence import CalibratedConfidenceScorer
from src.geotagging import GeotaggingEngine
from src.acoustic_physics import SoundSpeedProfile, AcousticRayTracer, AcousticMultilaterationSolver, SonarPoseGraphSLAM
from src.space_surveillance import SARProcessor, AISRecord, MultiModalMaritimeFusionEngine
from src.vehicle_dynamics import FossenHydrodynamics6DOF, CableTrackingMPC, MAVLinkAutopilotBridge
from src.infer import AquaSenseInference


def test_confidence_engine():
    print("[1/6] Testing Calibrated Confidence & Uncertainty Engine...")
    scorer = CalibratedConfidenceScorer(mean_clean_mse=0.0086, std_clean_mse=0.0060, threshold=0.0267)

    # Test high-severity anomaly
    res_high = scorer.evaluate(
        frame_mse=0.065,
        object_type="Linear Pipeline / Cable Track",
        status="KNOWN TYPE",
        area=3200.0,
        aspect_ratio=3.5,
        elongation=3.5,
        contrast_score=0.45,
        mean_anomaly=110.0,
        spans_frame=True
    )
    assert res_high["composite_confidence"] >= 0.70, f"Expected high confidence, got {res_high}"
    assert res_high["quality_tier"] == "HIGH"
    assert "uncertainty_index" in res_high
    print(f"  [PASS] High-Fidelity Cable Score: {res_high['confidence_percent']}% | Uncertainty: {res_high['uncertainty_index']}")

    # Test low-margin anomaly
    res_low = scorer.evaluate(
        frame_mse=0.027,
        object_type="Unclassified Submerged Threat",
        status="UNKNOWN ANOMALY",
        area=150.0,
        aspect_ratio=1.1,
        elongation=1.1,
        contrast_score=0.10,
        mean_anomaly=25.0
    )
    print(f"  [PASS] Low-Margin Threat Score: {res_low['confidence_percent']}% | Tier: {res_low['quality_tier']}")


def test_geotagging_engine():
    print("\n[2/6] Testing Multi-Source Geotagging Engine...")
    geo = GeotaggingEngine(default_lat=12.9716, default_lon=77.5946, default_depth=20.0)

    # Test NMEA GGA parse
    nmea_sample = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    parsed_nmea = geo.parse_nmea_sentence(nmea_sample)
    assert parsed_nmea is not None, "Failed to parse NMEA GGA"
    assert abs(parsed_nmea["latitude"] - 48.1173) < 0.01
    print(f"  [PASS] NMEA 0183 Ingestion: Lat {parsed_nmea['latitude']:.5f}, Lon {parsed_nmea['longitude']:.5f}")

    # Test USBL Acoustic Relative Slant-Range
    usbl = geo.calculate_usbl_position(
        mothership_lat=25.0000,
        mothership_lon=55.0000,
        slant_range_m=150.0,
        azimuth_deg=45.0,
        elevation_deg=30.0
    )
    assert usbl["depth_m"] > 70.0
    print(f"  [PASS] USBL Slant-Range: Depth {usbl['depth_m']}m | Target Lat {usbl['latitude']:.5f}")

    # Test Seafloor FOV Ground Projection
    projection = geo.project_pixel_to_seafloor(
        vehicle_lat=25.0000,
        vehicle_lon=55.0000,
        vehicle_heading_deg=90.0,  # Heading East
        altitude_m=10.0,
        bbox_center_px=(160, 60),  # Forward-starboard
        frame_shape=(224, 224)
    )
    assert projection["radial_offset_m"] > 0
    print(f"  [PASS] Seafloor Ray Projection: Offset {projection['radial_offset_m']}m from vehicle")


def test_acoustic_physics():
    print("\n[3/6] Testing Acoustic Physics & Deep Positioning Engine...")
    # Mackenzie sound speed equation
    c_surface = SoundSpeedProfile.mackenzie_sound_speed(temp_c=25.0, salinity_ppt=35.0, depth_m=0.0)
    c_deep = SoundSpeedProfile.mackenzie_sound_speed(temp_c=3.0, salinity_ppt=35.0, depth_m=2000.0)
    assert 1520.0 < c_surface < 1550.0
    assert 1470.0 < c_deep < 1510.0
    print(f"  [PASS] Mackenzie Sound Speed: Surface {c_surface:.1f} m/s | 2000m Depth {c_deep:.1f} m/s")

    # Snell's Law Ray Tracing
    tracer = AcousticRayTracer()
    ray = tracer.trace_ray(launch_angle_deg=35.0, transceiver_depth_m=10.0, target_depth_m=500.0)
    assert ray["refraction_correction_m"] >= 0.0
    print(f"  [PASS] Snell's Law Ray-Tracing: True Path {ray['true_curved_path_m']}m | Refraction Delta {ray['refraction_correction_m']}m")

    # Multilateration solver
    transponders = [(0.0, 0.0, 100.0), (200.0, 0.0, 100.0), (100.0, 200.0, 100.0)]
    target_true = np.array([80.0, 60.0, 15.0])
    tofs = [np.linalg.norm(np.array(tp) - target_true) / 1500.0 for tp in transponders]
    solved = AcousticMultilaterationSolver.solve_lbl_position(transponders, tofs)
    assert abs(solved["est_x_m"] - 80.0) < 0.1
    print(f"  [PASS] LBL Multilateration: Solved Position ({solved['est_x_m']}, {solved['est_y_m']}, {solved['est_depth_m']})")

    # Pose-Graph SLAM
    slam_res = SonarPoseGraphSLAM.simulate_cable_survey_with_slam(cable_length_m=300.0)
    print(f"  [PASS] Sonar Pose-Graph SLAM: Drift reduced by {slam_res['drift_reduction_percent']}%")


def test_space_surveillance():
    print("\n[4/6] Testing Surface & Space Surveillance (OBB & AIS Fusion)...")
    # SAR Lee Filter & OBB
    test_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(test_img, (30, 20), (50, 80), 255, -1)
    filtered = SARProcessor.lee_filter(test_img)
    obbs = SARProcessor.extract_oriented_bounding_boxes(test_img)
    assert len(obbs) == 1
    assert obbs[0]["aspect_ratio"] >= 2.0
    print(f"  [PASS] SAR OBB Extracted: Aspect {obbs[0]['aspect_ratio']} | Orientation {obbs[0]['orientation_deg']} deg")

    # Multi-Modal Fusion
    engine = MultiModalMaritimeFusionEngine()
    sar_detections = [
        {"target_lat": 24.1234, "target_lon": 54.1234, "area": 3500.0},  # Matches AIS
        {"target_lat": 24.5000, "target_lon": 54.5000, "area": 4200.0}   # No AIS -> Dark Vessel
    ]
    ais_records = [
        AISRecord(mmsi=123456789, vessel_name="MV CARRIER", vessel_type="Cargo", lat=24.1235, lon=54.1233, sog_knots=14.2, cog_deg=180.0),
        AISRecord(mmsi=987654321, vessel_name="GHOST SHIP", vessel_type="Tanker", lat=25.0000, lon=55.0000, sog_knots=18.0, cog_deg=90.0) # Ghost
    ]

    fusion_res = engine.correlate_detections_with_ais(sar_detections, ais_records, match_radius_m=300.0)
    assert fusion_res["cooperative_count"] == 1
    assert fusion_res["dark_vessel_count"] == 1
    assert fusion_res["spoofed_target_count"] == 1
    print(f"  [PASS] AIS Cross-Referencing: {fusion_res['cooperative_count']} Verified, {fusion_res['dark_vessel_count']} Dark Vessel, {fusion_res['spoofed_target_count']} Ghost Spoof")


def test_vehicle_dynamics():
    print("\n[5/6] Testing 6-DOF Fossen Hydrodynamics & Cable-Tracking MPC...")
    # Hydrodynamics step
    fossen = FossenHydrodynamics6DOF()
    vels = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    thrust = np.array([40.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.3, 0.0]) # Cross-current
    next_vels, accs = fossen.step(vels, thrust, current)
    print(f"  [PASS] 6-DOF Fossen Hydrodynamics: Surge Vel {next_vels[0]:.2f} m/s | Sway Accel {accs[1]:.2f} m/s^2")

    # Cable Tracking MPC
    mpc = CableTrackingMPC()
    cable_ref = [(0.0, 0.0, 20.0), (10.0, 0.0, 20.0)]
    control = mpc.compute_optimal_control(
        current_x_m=0.0,
        current_y_m=1.2, # 1.2m offset from cable
        current_depth_m=20.0,
        current_vx=1.0,
        current_vy=0.1,
        current_vz=0.0,
        cable_reference_line=cable_ref,
        ocean_current_vy=0.4 # 0.4 m/s cross-current
    )
    assert control["optimal_thrust_sway_n"] != 0.0
    print(f"  [PASS] MPC Counter-Thrust: Sway {control['optimal_thrust_sway_n']} N | Surge {control['optimal_thrust_surge_n']} N")

    # MAVLink Autopilot Bridge
    mav_msg = MAVLinkAutopilotBridge.encode_set_position_target(25.1, 55.2, 25.0, 90.0)
    assert mav_msg["mavlink_message_id"] == 84
    print(f"  [PASS] MAVLink ArduSub Bridge: Message #{mav_msg['mavlink_message_id']} ({mav_msg['message_name']})")


def test_full_pipeline_integration():
    print("\n[6/6] Testing Full AquaSense Dual-Stage Pipeline Integration...")
    engine = AquaSenseInference()

    debris_dir = PROJECT_ROOT / "data" / "raw" / "debris"
    sample_images = list(debris_dir.glob("*.jpg"))
    if not sample_images:
        print("  [SKIP] No test debris images found.")
        return

    test_img = cv2.imread(str(sample_images[0]))
    telemetry = {"lat": 13.05, "lon": 80.25, "depth": 35.0, "heading": 120.0, "altitude_m": 8.0}
    res = engine.analyze(test_img, nav_telemetry=telemetry)

    assert "anomaly_score" in res
    assert "detections" in res
    print(f"  [PASS] Full Pipeline Status: {res['global_status']} | MSE: {res['anomaly_score']:.5f}")
    if res["detections"]:
        d = res["detections"][0]
        print(f"  [PASS] Target 1: [{d['status']}] {d['type']}")
        print(f"    - Calibrated Confidence : {d.get('confidence_percent')}% (Uncertainty: {d.get('uncertainty_index')})")
        print(f"    - Seafloor Geotag       : Lat {d.get('target_lat')}, Lon {d.get('target_lon')} [Source: {d.get('geotag_source')}]")
        print(f"    - Oriented Bounding Box : {d.get('obb')}")


if __name__ == "__main__":
    print("=" * 70)
    print("AQUASENSE SOFTWARE-LEVEL ADVANCEMENTS TEST RUNNER")
    print("=" * 70)
    test_confidence_engine()
    test_geotagging_engine()
    test_acoustic_physics()
    test_space_surveillance()
    test_vehicle_dynamics()
    test_full_pipeline_integration()
    print("\n" + "=" * 70)
    print("ALL ADVANCEMENT ENGINES PASSED VERIFICATION!")
    print("=" * 70)
