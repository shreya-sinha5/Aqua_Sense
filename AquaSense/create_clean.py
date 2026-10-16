import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description="Extract clean background crops from sonar imagery")
parser.add_argument("--source", type=Path, default=Path("data/raw/source"), help="Path to source train directory")
parser.add_argument("--max-clean", type=int, default=300, help="Maximum number of clean samples to create")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = args.source if args.source.is_absolute() else PROJECT_DIR / args.source

IMAGE_DIR = SOURCE_DIR / "images"
LABEL_DIR = SOURCE_DIR / "labels"
CLEAN_DIR = PROJECT_DIR / "data" / "raw" / "clean"
MAX_CLEAN = args.max_clean

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
] if IMAGE_DIR.exists() else []

created = 0

for image_path in images:
    if created >= MAX_CLEAN:
        break

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    # Read all object boxes
    boxes = []

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            _, x_center, y_center, box_width, box_height = map(float, values)

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            boxes.append((x1, y1, x2, y2))

    if not boxes:
        continue

    # Try the four corners first because they are
    # less likely to contain the labelled object.
    candidates = [
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height)
    ]

    selected = None

    for x1, y1, x2, y2 in candidates:
        overlaps = False

        for bx1, by1, bx2, by2 in boxes:
            if not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2):
                overlaps = True
                break

        if not overlaps:
            selected = (x1, y1, x2, y2)
            break

    if selected is None:
        continue

    x1, y1, x2, y2 = selected

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # Resize crop to a consistent size
    crop = cv2.resize(crop, (224, 224))

    output_path = CLEAN_DIR / f"clean_{created + 1:04d}.jpg"

    cv2.imwrite(str(output_path), crop)

    created += 1


print("\nClean dataset creation complete!")
print("--------------------------------")
print(f"Clean images created: {created}")
print(f"Location: {CLEAN_DIR}")

if created == 0:
    print("\nWARNING: No clean images were created.")
else:
    print("\nYou can now train the autoencoder using these images.")