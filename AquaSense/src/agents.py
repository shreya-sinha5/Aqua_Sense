"""
AquaSense — Autonomous Multi-Agent Advisory Intelligence Engine

Implements three cooperative agent roles using structured rule-based reasoning
for automated mission guidance, risk assessment, and operator briefings.

Agent Architecture:
    Agent 1 — Tactical AUV Navigation Agent
    Agent 2 — Environmental Risk Assessment Agent
    Agent 3 — Mission Briefing Synthesis Agent

All agents are deterministic rule-based systems (no LLM/API dependency),
designed for offline operation on AUV edge hardware.
"""

import math
from datetime import datetime, timezone


class TacticalNavigationAgent:
    """
    Agent 1 — Tactical AUV Navigation Agent

    Evaluates anomaly severity and bounding box spatial coordinates to compute:
    - Simulated thruster vectors and heading directives
    - Target avoidance offsets
    - Recovery priority classification (P1/P2/P3)
    """

    def evaluate(self, mse_score, threshold, detections, current_lat, current_lon):
        """
        Generates navigation directives based on anomaly severity.

        Returns:
            dict: Navigation assessment with priority, heading, thruster vector,
                  avoidance offset, and recommended speed.
        """
        is_anomaly = mse_score > threshold
        severity_ratio = mse_score / threshold if threshold > 0 else 0

        if not is_anomaly:
            return {
                "priority": "P3 — ROUTINE SURVEY",
                "priority_code": "P3",
                "heading": "MAINTAIN GRID SEARCH PATTERN",
                "speed_knots": 2.0,
                "thruster_vector": [0.0, 0.0, 0.0],
                "avoidance_offset_m": 0.0,
                "hold_position": False,
                "rationale": (
                    "No anomalous signatures detected. Continue systematic "
                    "grid search at standard survey speed."
                )
            }

        # P1: Critical — immediate intervention required
        if severity_ratio > 1.5:
            # Compute simulated avoidance bearing from detection centroid
            avoidance_bearing = self._compute_avoidance_bearing(detections)
            return {
                "priority": "P1 — CRITICAL RECOVERY",
                "priority_code": "P1",
                "heading": "HOLD POSITION & ASCEND 5m — AWAIT OPERATOR",
                "speed_knots": 0.0,
                "thruster_vector": [0.0, 0.0, 1.0],  # Vertical ascent
                "avoidance_offset_m": 15.0,
                "hold_position": True,
                "rationale": (
                    f"MSE severity ratio {severity_ratio:.2f}x exceeds critical "
                    f"threshold (1.5x). Immediate position hold and vertical "
                    f"ascent recommended. Avoidance bearing: {avoidance_bearing:.1f}°."
                )
            }

        # P2: High priority — scheduled detailed mapping
        return {
            "priority": "P2 — SCHEDULED MAPPING",
            "priority_code": "P2",
            "heading": "REDUCE SPEED — INITIATE CLOSE-RANGE SURVEY",
            "speed_knots": 0.5,
            "thruster_vector": [0.3, 0.0, -0.2],  # Slow forward, slight descent
            "avoidance_offset_m": 5.0,
            "hold_position": False,
            "rationale": (
                f"Moderate anomaly detected (severity ratio: {severity_ratio:.2f}x). "
                f"Reduce to survey speed for close-range mapping pass. "
                f"{len(detections)} target(s) require detailed profiling."
            )
        }

    def _compute_avoidance_bearing(self, detections):
        """Computes simulated avoidance bearing from detection bounding box centroids."""
        if not detections:
            return 0.0

        # Use first detection's bounding box centroid to simulate bearing
        d = detections[0]
        if isinstance(d, dict) and "bbox" in d:
            bbox = d["bbox"]
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            # Map image coordinates to 0-360 bearing (simulated)
            bearing = (math.atan2(cy - 112, cx - 112) * 180 / math.pi + 360) % 360
            return bearing
        return 45.0  # Default bearing


class EnvironmentalRiskAgent:
    """
    Agent 2 — Environmental Risk Assessment Agent

    Assesses target geometry (area, aspect ratio) and detection labels to
    flag ecological hazards and environmental risk classifications.
    """

    # Risk classification rules based on detection characteristics
    HAZARD_RULES = {
        "entanglement": {
            "keywords": ["rope", "cable", "net", "line", "wire", "linear", "fishing"],
            "risk_level": "HIGH",
            "description": "Entanglement Hazard: Submerged Rope/Cable/Net Structure",
            "ecological_note": "Ghost net entanglement risk to marine megafauna (turtles, cetaceans)"
        },
        "structural": {
            "keywords": ["shipwreck", "hull", "metal", "structural", "pipeline", "wreck", "steel"],
            "risk_level": "MEDIUM",
            "description": "Structural Artifact: Heavy Hull/Metal Corrosion Detected",
            "ecological_note": "Potential leachate contamination from corroded metallic structures"
        },
        "aircraft": {
            "keywords": ["aircraft", "plane", "airframe", "fuselage", "wing"],
            "risk_level": "HIGH",
            "description": "Aircraft Wreckage: Aeronautical Debris Field",
            "ecological_note": "Possible fuel/lubricant contamination hazard"
        },
        "fauna": {
            "keywords": ["fauna", "marine", "fish", "coral", "biological", "organic", "life"],
            "risk_level": "LOW",
            "description": "Fauna Proximity: Low Environmental Hazard",
            "ecological_note": "Biological signature — no intervention required"
        }
    }

    def evaluate(self, mse_score, threshold, detections):
        """
        Generates ecological hazard assessment from detection data.

        Returns:
            dict: Environmental assessment with hazard list, risk index,
                  and ecological notes.
        """
        hazards = []
        risk_scores = []

        for det in detections:
            # Extract label and geometry
            if isinstance(det, dict):
                label = det.get("type", det.get("label", "Unknown")).lower()
                area = float(det.get("area", 0))
                aspect_ratio = float(det.get("aspect_ratio", 1.0))
            elif isinstance(det, (list, tuple)):
                label = str(det[0]).lower()
                area = float(det[1]) if len(det) > 1 else 0
                aspect_ratio = float(det[2]) if len(det) > 2 else 1.0
            else:
                label = str(det).lower()
                area = 0
                aspect_ratio = 1.0

            # Match against hazard rules
            matched = False
            for hazard_type, rule in self.HAZARD_RULES.items():
                if any(kw in label for kw in rule["keywords"]):
                    hazards.append({
                        "type": hazard_type,
                        "risk_level": rule["risk_level"],
                        "description": rule["description"],
                        "ecological_note": rule["ecological_note"],
                        "source_label": label,
                        "area": area,
                        "aspect_ratio": aspect_ratio
                    })
                    risk_scores.append(
                        {"HIGH": 85, "MEDIUM": 55, "LOW": 15}[rule["risk_level"]]
                    )
                    matched = True
                    break

            if not matched:
                # Geometry-based heuristic classification
                if aspect_ratio > 4.0:
                    # Very elongated → likely linear debris (rope/cable)
                    hazards.append({
                        "type": "linear_debris",
                        "risk_level": "MEDIUM",
                        "description": "Elongated Structure: Potential Linear Debris / Cable",
                        "ecological_note": "Linear geometry suggests possible abandoned equipment",
                        "source_label": label,
                        "area": area,
                        "aspect_ratio": aspect_ratio
                    })
                    risk_scores.append(55)
                elif area > 5000:
                    # Large mass → structural anomaly
                    hazards.append({
                        "type": "large_mass",
                        "risk_level": "MEDIUM",
                        "description": "Unclassified Large Mass: Potential Synthetic/Metal Anomaly",
                        "ecological_note": "Large unidentified structure requires operator verification",
                        "source_label": label,
                        "area": area,
                        "aspect_ratio": aspect_ratio
                    })
                    risk_scores.append(60)
                else:
                    hazards.append({
                        "type": "unclassified",
                        "risk_level": "LOW",
                        "description": "Minor Unclassified Object: Low Environmental Risk",
                        "ecological_note": "Small anomalous signature — monitor on next survey pass",
                        "source_label": label,
                        "area": area,
                        "aspect_ratio": aspect_ratio
                    })
                    risk_scores.append(20)

        if not hazards:
            hazards.append({
                "type": "clear",
                "risk_level": "NONE",
                "description": "Zero Environmental Threat Detected",
                "ecological_note": "Seabed environment nominal — no ecological concerns",
                "source_label": "N/A",
                "area": 0,
                "aspect_ratio": 0
            })
            risk_scores.append(0)

        # Compute aggregate Ecological Hazard Index (0-100)
        hazard_index = min(100, max(risk_scores)) if risk_scores else 0

        return {
            "hazards": hazards,
            "hazard_descriptions": [h["description"] for h in hazards],
            "hazard_index": hazard_index,
            "max_risk_level": max(
                [h["risk_level"] for h in hazards],
                key=lambda x: {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(x, 0)
            ),
            "ecological_notes": [h["ecological_note"] for h in hazards]
        }


class MissionBriefingAgent:
    """
    Agent 3 — Mission Briefing Synthesis Agent

    Synthesizes outputs from the Tactical Navigation Agent and Environmental
    Risk Agent into a concise, professional ROV Operator Briefing report.
    """

    def generate_briefing(
        self,
        nav_assessment,
        env_assessment,
        mse_score,
        threshold,
        current_lat,
        current_lon,
        total_detections,
        depth=25.0,
        **kwargs
    ):
        """
        Generates a comprehensive ROV Tactical Briefing.

        Returns:
            str: Formatted mission briefing text.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Hazard summary line
        hazard_lines = []
        for h in env_assessment["hazards"]:
            hazard_lines.append(f"  [{h['risk_level']}] {h['description']}")
        hazard_block = "\n".join(hazard_lines) if hazard_lines else "  [NONE] Clear environment"

        briefing = (
            f"╔══════════════════════════════════════════════════════╗\n"
            f"║         ROV TACTICAL BRIEFING — AquaSense AI        ║\n"
            f"╠══════════════════════════════════════════════════════╣\n"
            f"║  Timestamp   : {timestamp:<38s} ║\n"
            f"║  Location    : [{current_lat:.5f}°N, {current_lon:.5f}°E]{'':>13s} ║\n"
            f"║  Targets     : {total_detections} anomalous structure(s) isolated{'':>9s} ║\n"
            f"╠══════════════════════════════════════════════════════╣\n"
            f"║  NAVIGATION DIRECTIVE                                ║\n"
            f"║  Priority    : {nav_assessment['priority']:<38s} ║\n"
            f"║  Heading     : {nav_assessment['heading']:<38s} ║\n"
            f"║  Speed       : {nav_assessment['speed_knots']:.1f} knots{'':>33s} ║\n"
            f"║  Avoidance   : {nav_assessment['avoidance_offset_m']:.1f}m offset{'':>30s} ║\n"
            f"╠══════════════════════════════════════════════════════╣\n"
            f"║  ECOLOGICAL ASSESSMENT                               ║\n"
            f"║  Hazard Index: {env_assessment['hazard_index']}/100 ({env_assessment['max_risk_level']}){'':>25s} ║\n"
            f"╠══════════════════════════════════════════════════════╣\n"
            f"║  TELEMETRY SIGNATURE                                 ║\n"
            f"║  MSE Score   : {mse_score:.6f}{'':>32s} ║\n"
            f"║  Threshold   : {threshold:.6f}{'':>32s} ║\n"
            f"║  Status      : {'ANOMALY' if mse_score > threshold else 'NOMINAL'}{'':>38s} ║\n"
            f"╚══════════════════════════════════════════════════════╝"
        )

        return briefing


class AUVAgentSystem:
    """
    Orchestrator for the AquaSense Multi-Agent Advisory System.

    Coordinates three specialized agents to deliver automated mission guidance:
        1. Tactical AUV Navigation Agent
        2. Environmental Risk Assessment Agent
        3. Mission Briefing Synthesis Agent
    """

    def __init__(self):
        self.nav_agent = TacticalNavigationAgent()
        self.risk_agent = EnvironmentalRiskAgent()
        self.briefing_agent = MissionBriefingAgent()

    def run_mission_advisory(
        self,
        mse_score,
        threshold,
        detections,
        current_lat=12.9716,
        current_lon=77.5946,
        depth=25.0,
        **kwargs
    ):
        """
        Executes the full multi-agent advisory pipeline.

        Args:
            mse_score (float): Frame reconstruction MSE.
            threshold (float): Calibrated anomaly threshold.
            detections (list): Detection list (dicts or tuples).
            current_lat (float): AUV latitude.
            current_lon (float): AUV longitude.
            depth (float): AUV survey depth in meters.

        Returns:
            dict: {
                "nav_assessment": dict,    # Full navigation agent output
                "env_assessment": dict,    # Full environmental risk output
                "nav_directive": str,      # Shorthand heading directive
                "priority": str,           # Priority classification string
                "priority_code": str,      # P1/P2/P3 code
                "hazards": list,           # Hazard description strings
                "hazard_index": int,       # Ecological Hazard Index 0-100
                "briefing": str            # Full ROV tactical briefing text
            }
        """
        # Agent 1: Tactical Navigation
        nav_assessment = self.nav_agent.evaluate(
            mse_score, threshold, detections, current_lat, current_lon
        )

        # Agent 2: Environmental Risk
        env_assessment = self.risk_agent.evaluate(
            mse_score, threshold, detections
        )

        # Agent 3: Mission Briefing (synthesizes Agent 1 + 2)
        briefing = self.briefing_agent.generate_briefing(
            nav_assessment=nav_assessment,
            env_assessment=env_assessment,
            mse_score=mse_score,
            threshold=threshold,
            current_lat=current_lat,
            current_lon=current_lon,
            total_detections=len(detections),
            depth=depth
        )

        return {
            "nav_assessment": nav_assessment,
            "env_assessment": env_assessment,
            "nav_directive": nav_assessment["heading"],
            "priority": nav_assessment["priority"],
            "priority_code": nav_assessment["priority_code"],
            "hazards": env_assessment["hazard_descriptions"],
            "hazard_index": env_assessment["hazard_index"],
            "briefing": briefing
        }
