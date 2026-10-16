# 🌊 AquaSense: AI Model Training & Dataset Preparation Guidelines
### Dual-Stage Underwater Object Detection & Anomaly System Engineering Manual

---

## 1. Executive Summary & Engine Architecture

**AquaSense** operates an intelligent two-stage hybrid surveillance pipeline engineered specifically for seabed sonar (Side-Scan Sonar, Synthetic Aperture Sonar) and underwater optical imagery. Marine surveillance environments suffer from **severe class imbalance and scarce labeled ground-truth data** for rare debris, unexploded ordnance, and novel submerged threats.

To overcome this, AquaSense decouples anomaly detection from semantic identification:

```
                  Raw Sonar / Optical Frame
                             │
                             ▼
     ┌─────────────────────────────────────────────────┐
     │   STAGE 1: Unsupervised Anomaly Engine           │
     │   • Symmetrical ConvAutoencoder                 │
     │   • Clean seabed reconstruction                 │
     │   • Pixel difference map: R(x,y) = |I - Î|      │
     │   • Global MSE vs Calibrated (Mean + 3σ) Thresh │
     └───────────────────────┬─────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
      [SEABED CLEAR]                  [ANOMALY DETECTED]
     Baseline Verified                        │
                                              ▼
                    ┌──────────────────────────────────────────────────┐
                    │   STAGE 2: Hybrid Object & Type Engine           │
                    │   • JET Colormap Heatmap Generation              │
                    │   • Otsu Binary Residual Thresholding            │
                    │   • Noise Filter: Area A >= 100 px               │
                    │   • Feature Profiling: A, w, h, w/h, Contrast    │
                    │   • Geometric Decision Matrix                    │
                    └─────────────────┬────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
         [KNOWN TYPE]                             [UNKNOWN ANOMALY]
   • Shipwreck Structure                     • Unclassified Submerged Threat
   • Linear Pipeline / Cable                 • Novel Foreign Object
   • Marine Fauna / Object                   • Operator Escalation
   • Aircraft Component                                    │
                 │                                         │
                 ▼                                         ▼
          🟩 Green Box                              🟥 Red Box
```

---

## 2. Stage 1 — Unsupervised Anomaly Engine (ConvAutoencoder)

### A. Dataset Requirements (`data/raw/clean/`)
* **Purity Requirement:** Training must be conducted **strictly on normal, clean seabed imagery** containing natural sand, silt, ripple patterns, and ambient backscatter, with zero human-made debris, shipwrecks, pipes, or foreign targets.
* **Volume:** Aim for a minimum of **200–500 diverse clean sonar frames**.
* **Environmental Variety:** Ensure diverse water depths, acoustic frequencies, sensor angles, and natural seafloor textures (flat sediment, sand dunes, gravel beds).
* **Extraction Utility:** When mining public sonar datasets (e.g. Roboflow, Kaggle), use `create_clean.py` to extract confirmed background crops outside labeled bounding boxes.

### B. Preprocessing & Augmentation Pipeline
1. **Grayscale Conversion:** Convert single or multi-channel acoustic/optical captures to 1-channel Grayscale (`cv2.COLOR_BGR2GRAY`).
2. **Standardized Resolution:** Resize input to exactly $224 \times 224$ pixels via area interpolation (`cv2.INTER_AREA`).
3. **Normalization:** Scale pixel intensities to the unit range $[0.0, 1.0]$:
   $$I_{\text{norm}}(x,y) = \frac{I(x,y)}{255.0}$$
4. **Data Augmentations:** Apply light, domain-appropriate augmentations to prevent overfitting to specific ripple orientations:
   * Random Horizontal Flip ($p = 0.5$)
   * Random Vertical Flip ($p = 0.5$)
   * Subtle Contrast Adjustment (`ColorJitter(contrast=0.1)`)
   * *Avoid extreme rotations or noise injection that destroy seabed baseline frequencies.*

### C. Model Architecture (`ConvAutoencoder`)
The autoencoder employs symmetrical convolutional compression and transposed deconvolutional expansion:

* **Encoder:**
  * `Conv2d(1 -> 16, k=3, p=1)` -> `BatchNorm2d(16)` -> `ReLU` -> `MaxPool2d(2)` ($112 \times 112$)
  * `Conv2d(16 -> 32, k=3, p=1)` -> `BatchNorm2d(32)` -> `ReLU` -> `MaxPool2d(2)` ($56 \times 56$)
  * `Conv2d(32 -> 64, k=3, p=1)` -> `BatchNorm2d(64)` -> `ReLU` -> `MaxPool2d(2)` ($28 \times 28$)
  * `Conv2d(64 -> 128, k=3, p=1)` -> `BatchNorm2d(128)` -> `ReLU` -> `MaxPool2d(2)` ($14 \times 14$ latent representation)
* **Decoder:**
  * `ConvTranspose2d(128 -> 64, k=2, s=2)` -> `ReLU` ($28 \times 28$)
  * `ConvTranspose2d(64 -> 32, k=2, s=2)` -> `ReLU` ($56 \times 56$)
  * `ConvTranspose2d(32 -> 16, k=2, s=2)` -> `ReLU` ($112 \times 112$)
  * `ConvTranspose2d(16 -> 1, k=2, s=2)` -> `Sigmoid` ($224 \times 224$)

### D. Training Hyperparameters
* **Loss Function:** Mean Squared Error (MSE) Loss:
  $$\mathcal{L}(I, \hat{I}) = \frac{1}{H \times W} \sum_{x=1}^{H} \sum_{y=1}^{W} \left(I(x,y) - \hat{I}(x,y)\right)^2$$
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{learning rate} = 1 \times 10^{-3}$)
* **Batch Size:** 16
* **Epochs:** 20–50 epochs (train until validation loss stabilizes with $\Delta\mathcal{L}_{\text{val}} < 10^{-5}$)
* **Split:** 80% Training, 20% Validation (fixed random seed = 42)

### E. Threshold Calibration Mathematics
Pass the clean validation image set through the trained `ConvAutoencoder` and compute individual frame reconstruction errors:
$$\text{MSE}_i = \frac{1}{N} \sum \left(I_i - \hat{I}_i\right)^2$$

Calculate the sample mean $\mu_{\text{clean}}$ and standard deviation $\sigma_{\text{clean}}$:
$$\mu_{\text{clean}} = \frac{1}{K}\sum_{i=1}^{K} \text{MSE}_i, \quad \sigma_{\text{clean}} = \sqrt{\frac{1}{K}\sum_{i=1}^{K} (\text{MSE}_i - \mu_{\text{clean}})^2}$$

The global anomaly threshold is calibrated at the **$3\sigma$ upper confidence boundary**:
$$\text{Threshold} = \mu_{\text{clean}} + 3 \times \sigma_{\text{clean}}$$

The calibrated value is persisted to `models/threshold.json`:
```json
{
    "threshold": 0.026747,
    "method": "Mean(MSE_clean) + 3 * StdDev(MSE_clean)",
    "mean_validation_error": 0.008666,
    "std_validation_error": 0.006027,
    "multiplier": 3.0,
    "validation_images": 33
}
```

---

## 3. Stage 2 — Hybrid Object & Type Classification Engine

When Stage 1 computes the pixel-wise difference map:
$$R(x,y) = |I(x,y) - \hat{I}(x,y)|$$

Stage 2 executes isolated spatial contour profiling to categorize the anomaly.

### A. Contour Extraction & Filtering Pipeline
1. **Heatmap Generation:** Apply OpenCV JET colormap (`cv2.applyColorMap(residual, cv2.COLORMAP_JET)`).
2. **Adaptive Otsu Thresholding:** Apply Otsu binarization to the Gaussian-smoothed residual map (`cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`).
3. **Morphological Filtering:** Apply elliptical opening and closing kernels ($5 \times 5$) to dissolve acoustic speckle noise while solidifying target hulls.
4. **Noise Area Rejection:** Discard any contour with area $A < 100\text{ px}$.
5. **Residual Floor Check:** Require mean contour residual $\ge 20\text{ intensity}$ to eliminate ambient seabed texture fluctuations.

### B. Feature Vector Extraction
For each isolated contour, compute:
* **Contour Area ($A$):** True pixel area (`cv2.contourArea(c)`).
* **Bounding Rect ($w, h$):** Width and height of minimal enclosing upright rectangle.
* **Aspect Ratio ($w/h$):** Horizontal vs vertical elongation.
* **Elongation Ratio:** $\max(w,h) / \min(w,h)$ invariant to orientation.
* **Contrast Score ($C$):** Normalized mean intensity delta between target region and surrounding seabed background.

### C. Type Labeling Decision Matrix

| Target Classification | Geometric & Contrast Rules | Status | Box Color |
| :--- | :--- | :--- | :--- |
| **Large Pipeline / Cable Track** | Spans $\ge 70\%$ of frame width or height, Elongation $\ge 2.2$, intersects boundary edges | `KNOWN TYPE` | 🟩 **Vibrant Green** |
| **Shipwreck Frame / Linear Debris** | $A > 2500\text{ px}$ AND ($\text{Aspect Ratio} > 2.0$ OR $\text{Aspect Ratio} < 0.5$) | `KNOWN TYPE` | 🟩 **Vibrant Green** |
| **Submerged Aircraft Component** | $1200 \le A \le 2500\text{ px}$, Elongation $\ge 1.6$ or Contrast $> 0.20$ | `KNOWN TYPE` | 🟩 **Vibrant Green** |
| **Submerged Marine Object / Fauna** | $A < 500\text{ px}$ with high relative contrast ($C > 0.18$ or mean anomaly $> 40$) | `KNOWN TYPE` | 🟩 **Vibrant Green** |
| **Unknown Submerged Anomaly** | Any high reconstruction error region ($\text{MSE} > \text{Threshold}$) not matching known geometric rules | `UNKNOWN ANOMALY` | 🟥 **Urgent Red** |

---

## 4. Team Execution Workflows & CLI Runbook

### Step 1: Prepare Clean Seabed Imagery
```powershell
# Extract clean background crops from raw sonar datasets
.\venv\Scripts\python.exe create_clean.py
```

### Step 2: Train the ConvAutoencoder
```powershell
# Train symmetrical autoencoder on clean seabed frames
.\venv\Scripts\python.exe src\train.py
```
*Outputs: `models/aquasense_ae.pth` and `models/training_info.json`*

### Step 3: Calibrate Anomaly Threshold
```powershell
# Calculate Mean + 3 * StdDev on clean validation set
.\venv\Scripts\python.exe src\threshold.py
```
*Outputs: `models/threshold.json`*

### Step 4: Run Dual-Stage Automated Verification
```powershell
# Verify Stage 1 and Stage 2 against debris and clean test images
.\venv\Scripts\python.exe src\test_dual_stage.py
```

### Step 5: Launch AquaSense Mission Dashboard
```powershell
# Launch Streamlit web dashboard
.\venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

---

## 5. Defense Pitch Strategy & Presentation Guide

When presenting AquaSense to evaluators, mission commanders, or stakeholders, emphasize the following strategic points:

### 1. The Core Innovation: Solving Labeled Underwater Data Scarcity
> *"Traditional deep learning detectors like YOLO or Faster R-CNN require thousands of annotated bounding boxes for every single object category. In underwater mine warfare, shipwreck surveying, or deep-sea search and rescue, anomalies are rare and novel—operators cannot predict what shape a lost fuselage or unexploded ordnance will take.*
>
> *AquaSense solves this by training an **Unsupervised Reconstruction Autoencoder** strictly on ordinary seabed patterns. The network becomes a mathematical expert on clean ocean floor. Whenever any unexpected foreign object enters the sensor frame, the autoencoder fails to reconstruct it, creating a sharp residual spike. **This guarantees zero missed detections for unknown threats without requiring thousands of labeled samples.**"*

### 2. The Power of Dual-Stage Synergy: Actionable Clarity
> *"Once Stage 1 flags an anomaly, Stage 2 immediately extracts residual contour geometry, elongation, and acoustic contrast to classify known physical structures—distinguishing linear shipwreck frames, continuous undersea pipelines, and marine fauna, while explicitly labeling novel objects as **'Unknown Anomalies'** with urgent red brackets for human operator review.*
>
> *This eliminates alert fatigue while maintaining an uncompromised defensive safety perimeter."*

---

*Document Revision: 2.4 | AquaSense Core Engineering Team | Confidential & Proprietary*
