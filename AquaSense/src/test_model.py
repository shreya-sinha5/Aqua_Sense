from pathlib import Path
import json

import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import AquaSenseAutoencoder
from features import ShadowFeatureExtractor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
MODEL_PATH = PROJECT_DIR / "models" / "aquasense_ae.pth"
THRESHOLD_PATH = PROJECT_DIR / "models" / "threshold.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cpu")


with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

threshold = threshold_data["threshold"]


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

anomaly_class_index = dataset.class_to_idx["debris"]

anomaly_indices = [
    index
    for index, (_, label) in enumerate(dataset.samples)
    if label == anomaly_class_index
]

anomaly_dataset = torch.utils.data.Subset(
    dataset,
    anomaly_indices
)

loader = DataLoader(
    anomaly_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = AquaSenseAutoencoder().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

shadow_extractor = ShadowFeatureExtractor()

anomaly_count = 0
normal_count = 0
errors = []
shadow_scores = []


with torch.no_grad():

    for images, _ in loader:

        images = images.to(DEVICE)

        reconstructed = model(images)

        batch_errors = torch.mean(
            (images - reconstructed) ** 2,
            dim=(1, 2, 3)
        )

        for image_tensor, error in zip(
            images.cpu(),
            batch_errors.cpu()
        ):

            image_array = (
                image_tensor.squeeze().numpy() * 255
            ).clip(0, 255).astype("uint8")

            shadow_score = shadow_extractor.score(
                image_array
            )

            error_value = float(error.item())

            errors.append(error_value)
            shadow_scores.append(shadow_score)

            if error_value > threshold:
                anomaly_count += 1
            else:
                normal_count += 1


print()
print("AquaSense Model + Shadow Feature Test")
print("--------------------------------------")
print(f"Test images          : {len(errors)}")
print(f"Reconstruction threshold : {threshold:.6f}")
print(f"Detected as anomaly  : {anomaly_count}")
print(f"Detected as normal   : {normal_count}")

if errors:

    print(f"Minimum error        : {min(errors):.6f}")
    print(f"Maximum error        : {max(errors):.6f}")
    print(f"Average error        : {sum(errors) / len(errors):.6f}")

    detection_rate = (
        anomaly_count / len(errors)
    ) * 100

    print(f"Above-threshold rate : {detection_rate:.2f}%")

    print()
    print(
        f"Average shadow score : "
        f"{sum(shadow_scores) / len(shadow_scores):.4f}"
    )

print()
print("Shadow feature integration test complete.")