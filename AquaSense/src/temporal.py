from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent

SESSION_1_DIR = PROJECT_DIR / "data" / "sessions" / "session_1"
SESSION_2_DIR = PROJECT_DIR / "data" / "sessions" / "session_2"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "temporal"

IMAGE_SIZE = (224, 224)


def load_images(folder):
    image_files = sorted(
        [
            path
            for path in folder.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    images = []

    for path in image_files:
        image = cv2.imread(str(path))

        if image is not None:
            image = cv2.resize(image, IMAGE_SIZE)
            images.append((path.name, image))

    return images


def align_images(reference, current):
    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=500
    )

    keypoints_1, descriptors_1 = orb.detectAndCompute(
        reference_gray,
        None
    )

    keypoints_2, descriptors_2 = orb.detectAndCompute(
        current_gray,
        None
    )

    if descriptors_1 is None or descriptors_2 is None:
        return current, False, 0

    if len(descriptors_1) < 4 or len(descriptors_2) < 4:
        return current, False, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        descriptors_1,
        descriptors_2
    )

    matches = sorted(
        matches,
        key=lambda match: match.distance
    )

    if len(matches) < 4:
        return current, False, len(matches)

    source_points = np.float32(
        [
            keypoints_2[m.trainIdx].pt
            for m in matches[:50]
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            keypoints_1[m.queryIdx].pt
            for m in matches[:50]
        ]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        return current, False, len(matches)

    aligned = cv2.warpPerspective(
        current,
        homography,
        IMAGE_SIZE
    )

    inlier_count = 0

    if mask is not None:
        inlier_count = int(np.sum(mask))

    return aligned, True, inlier_count


def detect_change(reference, current):
    gray_1 = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    gray_2 = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        gray_1,
        gray_2
    )

    difference = cv2.GaussianBlur(
        difference,
        (5, 5),
        0
    )

    _, change_mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    changed_pixels = np.count_nonzero(
        change_mask
    )

    total_pixels = change_mask.size

    change_ratio = (
        changed_pixels / total_pixels
    )

    return change_mask, float(change_ratio)


def create_overlay(image, change_mask):
    overlay = image.copy()

    highlight = np.zeros_like(image)

    highlight[:, :, 2] = change_mask

    overlay = cv2.addWeighted(
        overlay,
        0.75,
        highlight,
        0.50,
        0
    )

    return overlay


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session_1 = load_images(
        SESSION_1_DIR
    )

    session_2 = load_images(
        SESSION_2_DIR
    )

    if not session_1:
        print("ERROR: No images found in session_1.")
        return

    if not session_2:
        print("ERROR: No images found in session_2.")
        return

    pair_count = min(
        len(session_1),
        len(session_2)
    )

    print()
    print("AquaSense Temporal Change Detection")
    print("------------------------------------")
    print(f"Session 1 images : {len(session_1)}")
    print(f"Session 2 images : {len(session_2)}")
    print(f"Pairs compared   : {pair_count}")
    print("Alignment        : ORB + Homography")
    print()

    aligned_count = 0
    detected_changes = 0

    for index in range(pair_count):
        name_1, image_1 = session_1[index]
        name_2, image_2 = session_2[index]

        aligned_image, aligned, inliers = align_images(
            image_1,
            image_2
        )

        if aligned:
            aligned_count += 1

        change_mask, change_ratio = detect_change(
            image_1,
            aligned_image
        )

        overlay = create_overlay(
            aligned_image,
            change_mask
        )

        status = (
            "CHANGE DETECTED"
            if change_ratio > 0.02
            else "NO SIGNIFICANT CHANGE"
        )

        if status == "CHANGE DETECTED":
            detected_changes += 1

        output_name = (
            f"comparison_{index + 1:04d}.jpg"
        )

        output_path = (
            OUTPUT_DIR / output_name
        )

        cv2.imwrite(
            str(output_path),
            overlay
        )

        alignment_status = (
            "ALIGNED"
            if aligned
            else "ALIGNMENT FAILED"
        )

        print(
            f"Pair {index + 1:02d} | "
            f"{name_1} vs {name_2} | "
            f"{alignment_status} | "
            f"Inliers: {inliers} | "
            f"Change ratio: {change_ratio:.4f} | "
            f"{status}"
        )

    print()
    print("Temporal analysis complete.")
    print("---------------------------")
    print(f"Pairs processed      : {pair_count}")
    print(f"Pairs aligned        : {aligned_count}")
    print(f"Changes detected     : {detected_changes}")
    print(f"Output folder        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()