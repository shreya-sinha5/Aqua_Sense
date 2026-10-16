# AquaSense — Autonomous Underwater Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00e5ff?style=for-the-badge)

**A dual-stage AI system for real-time underwater anomaly detection, sonar analysis, and AUV tactical intelligence.**

</div>

---

## 📌 Overview

AquaSense is a research-grade autonomous underwater vehicle (AUV) intelligence platform that combines **unsupervised deep learning**, **computer vision**, and **geospatial GIS** to detect, classify, and geolocate submerged anomalies — shipwrecks, pipelines, debris fields, and unidentified foreign objects — from sonar and optical underwater imagery.

It is built to operate in **GPS-denied, low-bandwidth subsea environments**, using acoustic telemetry compression, physics-informed preprocessing, and multi-agent advisory systems for mission decision support.

---

## 🎯 Key Features

### 🧠 Dual-Stage AI Detection Engine
- **Stage 1 — Convolutional Autoencoder (ConvAE):** Trained on clean seabed baselines; anomalies are flagged by reconstruction error (MSE) against a calibrated `Mean + 3σ` threshold
- **Stage 2 — Hybrid Object Profiler:** Residual contour geometry analysis classifies detected regions into known types (pipelines, shipwreck frames, aircraft components, marine fauna) or flags unknowns for operator review
- **Grad-CAM Saliency Maps:** Visual explainability overlays showing which pixels drove the anomaly decision

### 🗺️ Geospatial Tactical Command Center
- Interactive **Folium GIS map** with multi-layer tiles (OpenStreetMap, Esri Ocean Bathymetry, Tactical Dark Canvas)
- Real-time AUV position beacon with **georeferenced anomaly markers**
- **Anomaly heatmap overlay** across the survey operating theater
- OGC **GeoJSON export** compatible with QGIS, ArcGIS, and Google Earth

### 📡 Ultra-Low Bandwidth Acoustic Telemetry
- Encodes full detection results into compact **JSON acoustic packets** (typically < 200 bytes vs. 150 KB+ raw frame)
- Simulates **WHOI Micro-Modem / EvoLogics S2C** transmission at 80 bps
- Achieves >99% bandwidth reduction for subsea-to-surface uplink

### 🔬 Advanced Research Modules

| Module | Description |
|---|---|
| **Calibrated Confidence Scoring** | Epistemic uncertainty quantification replacing heuristic confidence values |
| **Acoustic Ray-Tracing** | Mackenzie SSP + Snell's Law refraction correction for depth-accurate positioning |
| **LBL Multilateration Solver** | Non-linear least squares positioning from 3 seabed transponders |
| **Sonar Pose-Graph SLAM** | IMU dead-reckoning drift elimination via loop closure (>95% drift reduction) |
| **SAR/AIS Maritime Fusion** | Lee speckle filtering + OBB vessel detection + dark vessel / AIS spoofing detection |
| **6-DOF Fossen Hydrodynamics** | Physics-accurate AUV dynamics model with ocean current disturbance rejection |
| **Cable-Tracking MPC** | Model Predictive Control for precise subsea cable-following missions |
| **MAVLink ArduSub Interface** | `SET_POSITION_TARGET_GLOBAL_INT` autopilot command encoding |

### 🖥️ Streamlit Tactical Dashboard
- **Cyber-oceanic dark HUD** with glassmorphism UI and animated telemetry beacons
- Real-time CPU/RAM edge hardware telemetry via `psutil`
- Multi-source navigation: AUV telemetry ray-projection, manual benchmark, NMEA 0183 parser, USBL acoustic slant-range
- **Forensic audit trail** — auto-archives every analyzed frame (original · heatmap · bounded detection) with CSV metadata
- Live camera feed with interval-based frame capture and session logging

---

## 🏗️ Architecture

```
AquaSense/
├── src/
│   ├── dashboard.py          # Main Streamlit tactical UI
│   ├── model.py              # ConvAutoencoder & AquaSenseAutoencoder architectures
│   ├── infer.py              # Dual-Stage Hybrid Inference Engine
│   ├── train.py              # Autoencoder training pipeline
│   ├── geotagging.py         # Multi-source geotagging & GeoJSON export
│   ├── acoustic_physics.py   # SSP, ray-tracing, LBL multilateration, SLAM
│   ├── agents.py             # Autonomous multi-agent tactical advisory system
│   ├── confidence.py         # Calibrated confidence & uncertainty quantification
│   ├── telemetry.py          # Acoustic micro-modem telemetry encoder
│   ├── space_surveillance.py # SAR processor + AIS multi-modal fusion engine
│   ├── vehicle_dynamics.py   # Fossen 6-DOF + Cable MPC + MAVLink bridge
│   ├── logger.py             # Forensic audit trail & event archiving
│   └── api.py                # FastAPI REST interface
├── models/
│   ├── aquasense_ae.pth      # Trained autoencoder weights
│   └── threshold.json        # Calibrated detection threshold metadata
├── data/
│   ├── samples/              # Benchmark sonar test images
│   └── processed/            # Detection CSVs & temporal results
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/shreya-sinha5/Aqua_Sense.git
cd Aqua_Sense/AquaSense
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run src/dashboard.py
```

Opens at **http://localhost:8501**

---

## 🌊 Detection Pipeline

```
Raw Sonar / Optical Frame
        │
        ▼
[Speckle Filter + CLAHE]          ← Physics-informed preprocessing
        │
        ▼
[Stage 1: ConvAutoencoder]        ← Learns clean seabed patterns
        │
        ├── MSE < Threshold  →  ✅ SEABED CLEAR
        │
        └── MSE ≥ Threshold  →  🚨 ANOMALY DETECTED
                │
                ▼
        [Stage 2: Hybrid Profiler]
                │
                ├── Contour geometry → Known type (pipeline, shipwreck…)
                └── No match         → Unknown anomaly (operator review)
                        │
                        ▼
              [Geotagging + Acoustic Telemetry + Forensic Archive]
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit >= 1.36` | Tactical dashboard UI |
| `torch >= 2.1` | Autoencoder inference & training |
| `opencv-python >= 4.8` | Image processing & contour analysis |
| `numpy >= 1.26` | Numerical computations |
| `pandas >= 2.0` | Detection logging & CSV management |
| `folium >= 0.16` | Interactive GIS map |
| `streamlit-folium >= 0.20` | Folium in Streamlit |
| `fastapi >= 0.110` | REST API interface |
| `psutil >= 5.9` | Edge hardware telemetry |

---

## 🧪 Testing

```bash
python src/test_advancements.py   # All 6 advancement module tests
python src/test_dual_stage.py     # Dual-stage detection pipeline
python src/test_object_detection.py  # Object detection & classification
```

---

## 🔌 REST API

```
POST /analyze           Submit image for dual-stage analysis
GET  /detections        Retrieve logged detection history
GET  /telemetry/latest  Get latest acoustic telemetry packet
GET  /health            System health check
```

Run with:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

---

## 📄 License

MIT License

---

## 👩‍💻 Author

**Shreya Sinha** — AUV Intelligence & Marine AI Research

---

<div align="center">
<sub>Built for deep-sea missions. Engineered for zero missed detections.</sub>
</div>
