from pathlib import Path
import json

import cv2
import numpy as np
import torch

from model import AquaSenseAutoencoder


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "aquasense_ae.pth"
THRESHOLD_PATH = PROJECT_DIR / "models" / "threshold.json"

INPUT_DIR = PROJECT_DIR / "data" / "raw" / "debris"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "heatmaps"

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")


def load_model():
    model = AquaSenseAutoencoder().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    normalized = gray.astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        normalized
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


def create_heatmap(model, image):
    tensor = preprocess(image)

    with torch.no_grad():
        reconstructed = model(tensor)

    original = tensor.squeeze().cpu().numpy()
    reconstruction = reconstructed.squeeze().cpu().numpy()

    difference = np.abs(
        original - reconstruction
    )

    difference = (
        difference * 255
    ).clip(0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        difference,
        cv2.COLORMAP_JET
    )

    original_color = cv2.cvtColor(
        (original * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        original_color,
        0.65,
        heatmap,
        0.35,
        0
    )

    reconstruction_error = float(
        np.mean(
            (original - reconstruction) ** 2
        )
    )

    return heatmap, overlay, reconstruction_error


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(THRESHOLD_PATH, "r") as file:
        threshold_data = json.load(file)

    threshold = threshold_data["threshold"]

    model = load_model()

    image_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(
            INPUT_DIR.glob(extension)
        )

    image_files = sorted(image_files)

    if not image_files:
        print("No input images found.")
        return

    print()
    print("AquaSense Explainability Heatmap")
    print("---------------------------------")
    print(f"Input images : {len(image_files)}")
    print(f"Threshold    : {threshold:.6f}")
    print()

    processed = 0

    for image_path in image_files:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Skipped unreadable image: "
                f"{image_path.name}"
            )
            continue

        heatmap, overlay, error = create_heatmap(
            model,
            image
        )

        status = (
            "ANOMALY"
            if error > threshold
            else "NORMAL"
        )

        output_name = (
            image_path.stem +
            "_heatmap.jpg"
        )

        output_path = (
            OUTPUT_DIR / output_name
        )

        cv2.imwrite(
            str(output_path),
            overlay
        )

        processed += 1

        print(
            f"{image_path.name} | "
            f"Error: {error:.6f} | "
            f"{status}"
        )

    print()
    print("Heatmap generation complete.")
    print("----------------------------")
    print(f"Images processed : {processed}")
    print(f"Output folder    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()