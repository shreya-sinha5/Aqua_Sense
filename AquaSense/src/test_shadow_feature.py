from pathlib import Path

import cv2
import numpy as np

from features import ShadowFeatureExtractor


PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "raw" / "clean"
DEBRIS_DIR = PROJECT_DIR / "data" / "raw" / "debris"


extractor = ShadowFeatureExtractor()


def get_images(folder):
    files = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(folder.glob(extension))

    return sorted(files)


def calculate_scores(image_files):
    scores = []

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        score = extractor.score(image)
        scores.append(score)

    return scores


clean_files = get_images(CLEAN_DIR)
debris_files = get_images(DEBRIS_DIR)

clean_scores = calculate_scores(clean_files)
debris_scores = calculate_scores(debris_files)


print()
print("AquaSense Shadow Feature Analysis")
print("----------------------------------")

print(f"Clean images tested  : {len(clean_scores)}")
print(f"Debris images tested : {len(debris_scores)}")

if clean_scores:
    print()
    print("Clean shadow scores")
    print(f"Minimum : {min(clean_scores):.4f}")
    print(f"Maximum : {max(clean_scores):.4f}")
    print(f"Average : {np.mean(clean_scores):.4f}")
    print(f"95th %  : {np.percentile(clean_scores, 95):.4f}")

if debris_scores:
    print()
    print("Debris shadow scores")
    print(f"Minimum : {min(debris_scores):.4f}")
    print(f"Maximum : {max(debris_scores):.4f}")
    print(f"Average : {np.mean(debris_scores):.4f}")
    print(f"95th %  : {np.percentile(debris_scores, 95):.4f}")

print()
print("Analysis complete.")