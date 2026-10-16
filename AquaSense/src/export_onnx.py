"""Export the trained AquaSense autoencoder to ONNX.

Run from the project root:
    python src/export_onnx.py
"""
from pathlib import Path
import torch
from model import AquaSenseAutoencoder

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / 'models' / 'aquasense_ae.pth'
OUT = BASE / 'models' / 'aquasense_ae.onnx'

model = AquaSenseAutoencoder()
model.load_state_dict(torch.load(MODEL, map_location='cpu', weights_only=True))
model.eval()
example = torch.randn(1, 1, 224, 224)

torch.onnx.export(
    model, example, OUT,
    input_names=['image'], output_names=['reconstruction'],
    dynamic_axes={'image': {0: 'batch'}, 'reconstruction': {0: 'batch'}},
    opset_version=17,
)
print(f'ONNX model exported to: {OUT}')
