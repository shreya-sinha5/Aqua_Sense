# AquaSense Upgrade

The dashboard now includes:

- Imported-image and webcam analysis.
- Object screening labels: **Marine Life / Seafloor**, **Unclassified Debris**, and **Major Debris — Metal / Structural**.
- Major-debris alert queue and bypass for likely marine-life/seafloor signatures.
- Object metadata: depth, latitude, longitude, geo coordinates, highlight ratio and visual material estimate.
- Persistent detailed detection history in `data/processed/detections.csv`.
- AI reconstruction display from the existing autoencoder.
- Grad-CAM-style explainability overlay for the autoencoder reconstruction objective.
- Glassmorphism / professional operator UI.
- Runtime integration/status hooks for NumPy, ONNX Runtime, NVIDIA TensorRT and OpenDrift.
- `src/export_onnx.py` to export the trained PyTorch autoencoder to ONNX.
- `src/deployment.py` for optional ONNX inference and TensorRT availability.

## Important model limitation

The supplied model is an **anomaly-detection autoencoder**, not a semantic object detector, spectrometer or radiation detector. Therefore the dashboard uses conservative screening labels and explicitly marks material composition as a visual estimate. Claims such as "atomic material" must be confirmed with a dedicated radiation/material sensor.

## Run

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

For ONNX export:

```bash
python src/export_onnx.py
```

OpenDrift requires environmental forcing/readers to produce a physically meaningful trajectory forecast; the UI therefore does not invent current/wind data.
