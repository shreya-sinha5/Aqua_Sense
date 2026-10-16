"""Optional accelerated inference backends.

The dashboard keeps the existing PyTorch model as the safe default. If an
ONNX model exists and onnxruntime is installed, ONNX inference can be used.
TensorRT is exposed as a deployment hook because building an engine is
GPU/CUDA/version specific and should be done on the target NVIDIA machine.
"""
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
ONNX_PATH = BASE / 'models' / 'aquasense_ae.onnx'


def onnx_available():
    try:
        import onnxruntime  # noqa: F401
        return ONNX_PATH.exists()
    except Exception:
        return False


def onnx_reconstruct(image_tensor):
    import onnxruntime as ort
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: image_tensor.astype(np.float32)})[0]
    return result


def tensorrt_available():
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False
