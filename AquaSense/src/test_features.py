from pathlib import Path

import cv2

from features import ShadowFeatureExtractor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEBRIS_DIR = PROJECT_DIR / "data" / "raw" / "debris"


image_files = list(DEBRIS_DIR.glob("*.jpg"))

if not image_files:
    image_files = list(DEBRIS_DIR.glob("*.png"))

if not image_files:
    raise FileNotFoundError("No images found in the debris dataset.")


image_path = image_files[0]

image = cv2.imread(str(image_path))

extractor = ShadowFeatureExtractor()

features = extractor.extract(image)
score = extractor.score(image)


print("AquaSense Shadow Feature Test")
print("-----------------------------")
print(f"Image: {image_path.name}")
print(f"Mean intensity : {features['mean_intensity']:.4f}")
print(f"Std intensity  : {features['std_intensity']:.4f}")
print(f"Shadow ratio   : {features['shadow_ratio']:.4f}")
print(f"Feature score  : {score:.4f}")
print()
print("Shadow feature extraction successful!")