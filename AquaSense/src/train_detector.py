import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset root (configurable via environment or default to local data/yolo_dataset)
import os
default_dataset = PROJECT_ROOT / "data" / "yolo_dataset"
DATASET_ROOT = Path(os.environ.get("AQUASENSE_DATASET_ROOT", str(default_dataset)))

TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"

VALID_IMAGES = DATASET_ROOT / "valid" / "images"
VALID_LABELS = DATASET_ROOT / "valid" / "labels"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR / "aquasense_detector.pth"
)


# Model configuration

CLASS_NAMES = [
    "background",
    "aircraft",
    "fish",
    "other",
    "shipwreck"
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.0005


# Dataset

class SonarDetectionDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir
    ):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = sorted(
            [
                path
                for path in self.image_dir.iterdir()
                if path.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        image = torch.from_numpy(
            __import__("numpy").array(image)
        )

        image = image.permute(
            2,
            0,
            1
        ).float() / 255.0

        boxes = []
        labels = []

        if label_path.exists():

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    values = (
                        line.strip().split()
                    )

                    if len(values) != 5:
                        continue

                    class_id = int(
                        values[0]
                    )

                    center_x = float(
                        values[1]
                    )

                    center_y = float(
                        values[2]
                    )

                    box_width = float(
                        values[3]
                    )

                    box_height = float(
                        values[4]
                    )

                    # YOLO normalized coordinates
                    x_center = center_x * width
                    y_center = center_y * height

                    box_w = box_width * width
                    box_h = box_height * height

                    x1 = (
                        x_center - box_w / 2
                    )

                    y1 = (
                        y_center - box_h / 2
                    )

                    x2 = (
                        x_center + box_w / 2
                    )

                    y2 = (
                        y_center + box_h / 2
                    )

                    # Keep boxes inside image
                    x1 = max(
                        0,
                        min(x1, width)
                    )

                    y1 = max(
                        0,
                        min(y1, height)
                    )

                    x2 = max(
                        0,
                        min(x2, width)
                    )

                    y2 = max(
                        0,
                        min(y2, height)
                    )

                    if x2 <= x1:
                        continue

                    if y2 <= y1:
                        continue

                    boxes.append(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    )

                    # Dataset classes:
                    # 0 aircraft
                    # 1 fish
                    # 2 other
                    # 3 shipwreck
                    #
                    # Faster R-CNN reserves:
                    # 0 = background
                    #
                    # Therefore add 1.
                    labels.append(
                        class_id + 1
                    )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        area = torch.zeros(
            (len(boxes),),
            dtype=torch.float32
        )

        if len(boxes) > 0:

            area = (
                (boxes[:, 2] - boxes[:, 0])
                *
                (boxes[:, 3] - boxes[:, 1])
            )

        iscrowd = torch.zeros(
            (len(boxes),),
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [index]
            ),
            "area": area,
            "iscrowd": iscrowd
        }

        return image, target


# Collate function

def collate_fn(batch):

    return tuple(
        zip(*batch)
    )


# Model creation

def create_model():

    print()
    print(
        "Loading Faster R-CNN MobileNetV3..."
    )

    weights = (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    )

    model = (
        fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights
        )
    )

    # Replace the original COCO classifier
    # with our 4-class AquaSense detector.
    in_features = (
        model.roi_heads.box_predictor.cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model


# Training pipeline

def train():

    print()
    print("=" * 65)
    print("AquaSense Sonar Object Detector")
    print("=" * 65)

    # Check dataset

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    if not TRAIN_IMAGES.exists():

        print()
        print(
            "ERROR: Training image folder not found:"
        )

        print(TRAIN_IMAGES)

        return

    if not TRAIN_LABELS.exists():

        print()
        print(
            "ERROR: Training label folder not found:"
        )

        print(TRAIN_LABELS)

        return

    # Load dataset

    dataset = SonarDetectionDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    print()
    print(
        f"Training images: {len(dataset)}"
    )

    if len(dataset) == 0:

        print(
            "ERROR: No training images found."
        )

        return

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Device

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cpu":

        print(
            "CPU training selected."
        )

        print(
            "This may take some time."
        )

    # Model

    model = create_model()

    model.to(device)

    # Optimizer

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE
    )

    # Training loop

    print()
    print(
        "Classes:"
    )

    for index, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{index}: {name}"
        )

    print()
    print(
        "Starting training..."
    )
    print()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        batch_count = 0

        for batch_index, (
            images,
            targets
        ) in enumerate(loader):

            images = [
                image.to(device)
                for image in images
            ]

            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                }
                for target in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(
                images,
                targets
            )

            total_loss_value = sum(
                loss
                for loss in loss_dict.values()
            )

            total_loss_value.backward()

            optimizer.step()

            total_loss += (
                total_loss_value.item()
            )

            batch_count += 1

            if (
                batch_index + 1
            ) % 10 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Batch "
                    f"{batch_index + 1}/{len(loader)} "
                    f"| Loss "
                    f"{total_loss_value.item():.4f}"
                )

        average_loss = (
            total_loss / batch_count
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"completed "
            f"| Average Loss: "
            f"{average_loss:.6f}"
        )
        print()

    # Save model

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            CLASS_NAMES,

        "num_classes":
            NUM_CLASSES
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print()
    print("=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print()
    print(
        "Model saved to:"
    )

    print(MODEL_PATH)

    print()


if __name__ == "__main__":

    try:

        train()

    except KeyboardInterrupt:

        print()
        print(
            "Training stopped by user."
        )

    except Exception as error:

        print()
        print(
            "TRAINING ERROR:"
        )

        print(error)

        print()
        print(
            "If the error is related to "
            "pretrained weights or downloading, "
            "send me the complete error."
        )