import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description="Prepare AquaSense dataset into clean and debris folders")
parser.add_argument("--source", type=Path, default=Path("data/raw/source"), help="Path to source train directory")
parser.add_argument("--max-clean", type=int, default=300, help="Maximum number of clean samples to copy")
parser.add_argument("--max-anomaly", type=int, default=300, help="Maximum number of anomaly samples to copy")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = args.source if args.source.is_absolute() else PROJECT_DIR / args.source

IMAGE_DIR = SOURCE / "images"
LABEL_DIR = SOURCE / "labels"

CLEAN_DIR = PROJECT_DIR / "data" / "raw" / "clean"
ANOMALY_DIR = PROJECT_DIR / "data" / "raw" / "debris"

MAX_CLEAN = args.max_clean
MAX_ANOMALY = args.max_anomaly

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
] if IMAGE_DIR.exists() else []

label_stems = {
    p.stem
    for p in LABEL_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".txt"
} if LABEL_DIR.exists() else set()

clean_images = []
anomaly_images = []

for image in images:
    if image.stem in label_stems:
        anomaly_images.append(image)
    else:
        clean_images.append(image)

clean_images = clean_images[:MAX_CLEAN]
anomaly_images = anomaly_images[:MAX_ANOMALY]

for image in clean_images:
    shutil.copy2(image, CLEAN_DIR / image.name)

for image in anomaly_images:
    shutil.copy2(image, ANOMALY_DIR / image.name)

print("\nAquaSense dataset preparation complete!")
print("----------------------------------------")
print(f"Total images found       : {len(images)}")
print(f"Images with labels       : {len(anomaly_images)}")
print(f"Images without labels    : {len(clean_images)}")
print()
print(f"Clean images copied      : {len(clean_images)}")
print(f"Anomaly images copied    : {len(anomaly_images)}")
print()
print(f"Clean folder             : {CLEAN_DIR}")
print(f"Anomaly folder           : {ANOMALY_DIR}")