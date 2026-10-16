"""Optional runtime integrations for AquaSense.

OpenDrift, ONNX Runtime and NVIDIA TensorRT are intentionally optional so
the Streamlit demo can run on a normal laptop. When installed, the dashboard
can report their availability and use ONNX/TensorRT backends for future
deployed models.

OpenDrift is exposed as a trajectory helper for geospatial tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np


@dataclass
class RuntimeStatus:
    numpy: str
    onnxruntime: str
    tensorrt: str
    opendrift: str


def _module_available(module_name: str) -> bool:
    """Check whether an optional Python module is installed."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def runtime_status() -> RuntimeStatus:
    """Return the availability of optional AquaSense runtimes."""

    onnxruntime_status = (
        "READY"
        if _module_available("onnxruntime")
        else "OPTIONAL / NOT INSTALLED"
    )

    tensorrt_status = (
        "READY"
        if _module_available("tensorrt")
        else "OPTIONAL / NOT INSTALLED"
    )

    opendrift_status = (
        "READY"
        if _module_available("opendrift")
        else "OPTIONAL / NOT INSTALLED"
    )

    return RuntimeStatus(
        numpy=np.__version__,
        onnxruntime=onnxruntime_status,
        tensorrt=tensorrt_status,
        opendrift=opendrift_status,
    )


def create_opendrift_note(latitude: float, longitude: float) -> str:
    """Return a deployment note for an OpenDrift trajectory origin.

    This deliberately does not invent ocean-current data. A real OpenDrift
    simulation needs a configured reader/forcing dataset.
    """

    if _module_available("opendrift"):
        return (
            f"OpenDrift available — trajectory origin can be seeded at "
            f"{latitude:.5f}, {longitude:.5f}. Configure environmental "
            "forcing before running a physical drift forecast."
        )

    return (
        "OpenDrift adapter ready; install and configure forcing data "
        "for real drift forecasts."
    )