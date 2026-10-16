import json
from pathlib import Path
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import ConvAutoencoder, AquaSenseAutoencoder


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
MODEL_PATH = PROJECT_DIR / "models" / "aquasense_ae.pth"
THRESHOLD_PATH = PROJECT_DIR / "models" / "threshold.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
VALIDATION_RATIO = 0.2
DEVICE = torch.device("cpu")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

clean_class_index = dataset.class_to_idx["clean"]

clean_indices = [
    index
    for index, (_, label) in enumerate(dataset.samples)
    if label == clean_class_index
]

clean_dataset = torch.utils.data.Subset(dataset, clean_indices)

validation_size = int(len(clean_dataset) * VALIDATION_RATIO)
training_size = len(clean_dataset) - validation_size

generator = torch.Generator().manual_seed(42)

_, validation_dataset = random_split(
    clean_dataset,
    [training_size, validation_size],
    generator=generator
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = ConvAutoencoder().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


errors = []

with torch.no_grad():
    for images, _ in validation_loader:
        images = images.to(DEVICE)

        reconstructed = model(images)

        batch_errors = torch.mean(
            (images - reconstructed) ** 2,
            dim=(1, 2, 3)
        )

        errors.extend(batch_errors.cpu().tolist())


errors = np.array(errors, dtype=np.float64)

mean_error = float(np.mean(errors))
std_error = float(np.std(errors))

# Master Prompt formula: Threshold = Mean(MSE_clean) + 3 * StdDev(MSE_clean)
threshold = float(mean_error + 3.0 * std_error)

threshold_data = {
    "threshold": threshold,
    "method": "Mean(MSE_clean) + 3 * StdDev(MSE_clean)",
    "mean_validation_error": mean_error,
    "std_validation_error": std_error,
    "multiplier": 3.0,
    "validation_images": int(len(errors)),
    "min_validation_error": float(np.min(errors)),
    "max_validation_error": float(np.max(errors))
}

with open(THRESHOLD_PATH, "w") as file:
    json.dump(threshold_data, file, indent=4)


print()
print("AquaSense Anomaly Threshold Calibration")
print("=======================================")
print(f"Validation clean images  : {len(errors)}")
print(f"Mean validation error    : {mean_error:.6f}")
print(f"StdDev validation error  : {std_error:.6f}")
print(f"Calibrated Threshold     : {threshold:.6f}")
print("Formula                  : Mean(MSE_clean) + 3 * StdDev(MSE_clean)")
print()
print(f"Saved to: {THRESHOLD_PATH}")
