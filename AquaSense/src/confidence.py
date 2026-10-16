"""
AquaSense — Calibrated Confidence Scoring & Uncertainty Quantification Engine

Transforms heuristic/hardcoded detection values into statistically principled,
multi-modal confidence scores.

Combines:
1. Reconstruction Z-score relative to clean seabed baseline (Mean + StdDev).
2. Contrast / Signal-to-Noise Ratio (SNR) delta against surrounding background.
3. Geometric & Morphological fidelity (aspect ratio, compactness, boundary validity).
4. Physics-inspired acoustic shadow contrast score.
"""

import math
from typing import Dict, Any, Optional
import numpy as np


class CalibratedConfidenceScorer:
    """
    Computes calibrated confidence metrics and epistemic uncertainty
    for detected underwater anomalies and classified structures.
    """

    def __init__(
        self,
        mean_clean_mse: float = 0.008666,
        std_clean_mse: float = 0.006027,
        threshold: float = 0.026747,
        k_sigmoid: float = 1.2
    ):
        self.mean_clean = max(1e-7, mean_clean_mse)
        self.std_clean = max(1e-7, std_clean_mse)
        self.threshold = threshold
        self.k_sigmoid = k_sigmoid

    def compute_reconstruction_probability(self, frame_mse: float) -> float:
        """
        Calculates calibrated anomaly probability P(anomaly | MSE) via
        logistic sigmoid of the standard score (Z-score) centered at threshold.
        """
        z_score = (frame_mse - self.threshold) / self.std_clean
        # Bound z_score to avoid overflow
        z_bounded = max(-15.0, min(15.0, z_score))
        prob = 1.0 / (1.0 + math.exp(-self.k_sigmoid * z_bounded))
        return float(np.clip(prob, 0.01, 0.999))

    def compute_snr_score(self, contrast_score: float, mean_anomaly: float) -> float:
        """
        Evaluates acoustic/optical target saliency against surrounding ambient seabed.
        """
        # Contrast score is in [0, 1], mean_anomaly in [0, 255]
        normalized_anomaly = min(1.0, mean_anomaly / 100.0)
        snr = 0.6 * min(1.0, contrast_score * 3.5) + 0.4 * normalized_anomaly
        return float(np.clip(snr, 0.05, 0.99))

    def compute_geometric_purity(
        self,
        object_type: str,
        area: float,
        aspect_ratio: float,
        elongation: float,
        spans_frame: bool
    ) -> float:
        """
        Scores how well the contour features match canonical physical geometry.
        """
        if "Linear Pipeline" in object_type or "Cable Track" in object_type:
            # High elongation and boundary spanning indicate high structural fidelity
            elong_score = min(1.0, (elongation - 1.5) / 2.0) if elongation >= 1.5 else 0.3
            span_score = 1.0 if spans_frame else 0.5
            purity = 0.6 * elong_score + 0.4 * span_score

        elif "Shipwreck" in object_type:
            # Large area with distinct directional aspect ratio
            area_score = min(1.0, area / 4000.0)
            aspect_dev = abs(aspect_ratio - 1.0)
            aspect_score = min(1.0, aspect_dev / 1.5)
            purity = 0.5 * area_score + 0.5 * aspect_score

        elif "Aircraft" in object_type:
            # Medium-large area with moderate elongation
            area_score = 1.0 - min(1.0, abs(area - 1800.0) / 1200.0)
            elong_score = min(1.0, elongation / 2.0)
            purity = 0.5 * area_score + 0.5 * elong_score

        elif "Marine Object" in object_type or "Fauna" in object_type:
            # Compact smaller object
            size_score = 1.0 if area < 500 else max(0.2, 1.0 - (area - 500) / 1000.0)
            purity = size_score

        else:  # Unclassified anomaly
            # General anomaly confidence is determined by statistical residual
            purity = 0.70

        return float(np.clip(purity, 0.1, 0.98))

    def evaluate(
        self,
        frame_mse: float,
        object_type: str,
        status: str,
        area: float,
        aspect_ratio: float,
        elongation: float,
        contrast_score: float,
        mean_anomaly: float,
        spans_frame: bool = False,
        shadow_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Full calibrated confidence evaluation producing composite score
        and itemized uncertainty breakdown.
        """
        p_recon = self.compute_reconstruction_probability(frame_mse)
        snr_score = self.compute_snr_score(contrast_score, mean_anomaly)
        geom_score = self.compute_geometric_purity(
            object_type, area, aspect_ratio, elongation, spans_frame
        )

        shadow_factor = 0.5 if shadow_score is None else float(np.clip(shadow_score, 0.0, 1.0))

        # Composite weighting
        if status == "KNOWN TYPE":
            composite = (
                0.40 * p_recon +
                0.35 * geom_score +
                0.15 * snr_score +
                0.10 * shadow_factor
            )
        else:  # UNKNOWN ANOMALY
            # Prioritize statistical anomaly certainty and SNR
            composite = (
                0.60 * p_recon +
                0.25 * snr_score +
                0.15 * shadow_factor
            )

        composite = float(np.clip(composite, 0.05, 0.98))

        # Epistemic uncertainty is high when features conflict
        uncertainty = float(np.clip(
            abs(p_recon - geom_score) * 0.5 + (1.0 - snr_score) * 0.5,
            0.02, 0.95
        ))

        return {
            "composite_confidence": round(composite, 4),
            "confidence_percent": round(composite * 100.0, 1),
            "reconstruction_probability": round(p_recon, 4),
            "geometric_purity": round(geom_score, 4),
            "snr_score": round(snr_score, 4),
            "shadow_fidelity": round(shadow_factor, 4),
            "uncertainty_index": round(uncertainty, 4),
            "quality_tier": "HIGH" if composite >= 0.80 else ("MODERATE" if composite >= 0.50 else "LOW")
        }
