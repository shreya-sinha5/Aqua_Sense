from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import ConvAutoencoder, AquaSenseAutoencoder


PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "raw" / "clean"
MODEL_PATH = PROJECT_DIR / "models" / "aquasense_ae.pth"
INFO_PATH = PROJECT_DIR / "models" / "training_info.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.2

DEVICE = torch.device("cpu")


# Light data augmentations for sonar robustness: flips & minor contrast adjustments
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.1),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


raw_dataset = datasets.ImageFolder(
    CLEAN_DIR.parent
)

clean_class_index = raw_dataset.class_to_idx["clean"]

clean_indices = [
    index
    for index, (_, label) in enumerate(raw_dataset.samples)
    if label == clean_class_index
]

total_clean = len(clean_indices)
validation_size = int(total_clean * VALIDATION_RATIO)
training_size = total_clean - validation_size

generator = torch.Generator().manual_seed(42)
shuffled_indices = torch.randperm(total_clean, generator=generator).tolist()

train_indices = [clean_indices[i] for i in shuffled_indices[:training_size]]
val_indices = [clean_indices[i] for i in shuffled_indices[training_size:]]

# Build datasets with separate transforms
train_dataset_base = datasets.ImageFolder(CLEAN_DIR.parent, transform=train_transform)
val_dataset_base = datasets.ImageFolder(CLEAN_DIR.parent, transform=val_transform)

train_dataset = torch.utils.data.Subset(train_dataset_base, train_indices)
validation_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = ConvAutoencoder().to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("AquaSense ConvAutoencoder Training")
print("-----------------------------------")
print(f"Clean images       : {total_clean}")
print(f"Training images    : {training_size}")
print(f"Validation images  : {validation_size}")
print(f"Epochs             : {EPOCHS}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Learning rate      : {LEARNING_RATE}")
print(f"Device             : {DEVICE}")
print()



training_losses = []
validation_losses = []


for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for images, _ in train_loader:
        images = images.to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for images, _ in validation_loader:
            images = images.to(DEVICE)

            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            validation_loss += loss.item()

    validation_loss /= len(validation_loader)

    training_losses.append(train_loss)
    validation_losses.append(validation_loss)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {validation_loss:.6f}"
    )


torch.save(model.state_dict(), MODEL_PATH)


import json

training_info = {
    "clean_images": total_clean,
    "training_images": training_size,
    "validation_images": validation_size,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "final_training_loss": training_losses[-1],
    "final_validation_loss": validation_losses[-1]
}

with open(INFO_PATH, "w") as file:
    json.dump(training_info, file, indent=4)


print()
print("Training complete!")
print("------------------")
print(f"Model saved      : {MODEL_PATH}")
print(f"Training info    : {INFO_PATH}")
print(f"Final train loss : {training_losses[-1]:.6f}")
print(f"Final val loss   : {validation_losses[-1]:.6f}")