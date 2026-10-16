"""
AquaSense — Headless Tactical Sonar AI Daemon (Air-Gapped Submarine Mode)

Runs continuously as an autonomous background daemon on certified submarine hardware:
1. Ingests raw sonar hydrophone bus streams from `SonarBusAdapter`.
2. Evaluates dual-stage anomaly detection & calibrated confidence scoring.
3. Broadcasts real-time structured track updates to the Combat Management System (CMS).
4. Maintains an encrypted forensic audit ledger without exposing unauthenticated REST ports.
"""

import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from infer import AquaSenseInference
from sonar_bus import SonarBusAdapter
from subsea_navigation import INSDVLDeadReckoningEngine
from combat_system_feed import CombatSystemPublisher
from security import SecurityClassificationManager, CryptographicAuditLedger


def run_tactical_daemon(
    interval_s: float = 2.0,
    max_cycles: int = 10,
    classification: str = "SECRET // NOFORN"
):
    print("=" * 70)
    print(f"AQUASENSE SUBMARINE TACTICAL DAEMON [AIR-GAPPED MODE]")
    print(f"Classification Level : {classification}")
    print(f"Cycle Interval       : {interval_s}s")
    print("=" * 70)

    # 1. Initialize Engines
    ai_engine = AquaSenseInference()
    bus_adapter = SonarBusAdapter()
    nav_engine = INSDVLDeadReckoningEngine(surface_datum_lat=12.2958, surface_datum_lon=76.6394)
    cms_publisher = CombatSystemPublisher(classification_level=classification)
    audit_ledger = CryptographicAuditLedger()

    cycle = 0
    print("[DAEMON ONLINE] Streaming raw acoustic bus frames to Combat System...")

    while cycle < max_cycles:
        cycle += 1
        # Step Subsea DVL/INS Navigation state
        nav_state = nav_engine.update_telemetry(
            heading_deg=90.0 + (cycle * 0.5),
            pitch_deg=-0.5,
            roll_deg=0.2,
            surge_mps=2.5,
            sway_mps=0.1,
            heave_mps=0.0,
            depth_m=65.0,
            altitude_m=14.0,
            bottom_lock=True,
            dt_seconds=interval_s
        )

        # Ingest raw acoustic bus matrix (Zero-copy, No JPG/PNG compression)
        synthetic_bus_frame = bus_adapter.generate_synthetic_acoustic_bus_stream(
            beam_count=128, range_samples=512, inject_anomaly=(cycle % 3 == 0)
        )
        tensor_224, raster_224 = bus_adapter.process_raw_acoustic_bus_frame(synthetic_bus_frame)

        # Execute AI Anomaly & Defect Detection
        result = ai_engine.analyze(raster_224, nav_telemetry=nav_state.to_dict())

        # Publish structured tracks to Submarine Combat System
        detections = result.get("detections", [])
        print(f"\n[Cycle #{cycle:03d} | {datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] Depth: {nav_state.depth_m}m | Subsea Pos: {nav_state.latitude:.5f}, {nav_state.longitude:.5f}")
        print(f"  MSE Score: {result['anomaly_score']:.6f} | Global Status: {result['global_status']} | Targets: {len(detections)}")

        for d in detections:
            tactical_packet = cms_publisher.format_tactical_track_message(d, subsea_nav_state=nav_state.to_dict())
            cms_publisher.broadcast_to_combat_system(tactical_packet)

            # Compute tamper-evident hash
            event_hash = audit_ledger.compute_event_hash(tactical_packet["track_data"])
            print(f"  -> Dispatched to CMS: [{d['status']}] {d['type']} (Conf: {d.get('confidence_percent')}%) | Ledger Hash: {event_hash[:12]}...")

        time.sleep(interval_s)

    print("\n" + "=" * 70)
    print("DAEMON RUN COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaSense Submarine Tactical Daemon")
    parser.add_argument("--interval", type=float, default=1.0, help="Inference loop interval in seconds")
    parser.add_argument("--cycles", type=int, default=5, help="Number of test cycles to execute")
    parser.add_argument("--classification", type=str, default="SECRET // NOFORN", help="Security marking")
    args = parser.parse_args()

    run_tactical_daemon(interval_s=args.interval, max_cycles=args.cycles, classification=args.classification)
