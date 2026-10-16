from pathlib import Path
import csv
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import AquaSenseAutoencoder


PROJECT_DIR = Path(__file__).resolve().parent.parent

SESSION_1_DIR = PROJECT_DIR / "data" / "sessions" / "session_1"
SESSION_2_DIR = PROJECT_DIR / "data" / "sessions" / "session_2"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "temporal_ai"

MODEL_PATH = PROJECT_DIR / "models" / "aquasense_ae.pth"
THRESHOLD_PATH = PROJECT_DIR / "models" / "threshold.json"

RESULTS_CSV = OUTPUT_DIR / "temporal_detections.csv"

IMAGE_SIZE = 224
CHANGE_THRESHOLD = 0.02
MIN_CHANGE_AREA = 500
PADDING = 15


def load_model():
    model = AquaSenseAutoencoder()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model


def load_threshold():
    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:
        data = json.load(file)

    return float(data["threshold"])


def align_frames(reference, current):
    gray_reference = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    gray_current = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=500
    )

    keypoints_reference, descriptors_reference = (
        orb.detectAndCompute(
            gray_reference,
            None
        )
    )

    keypoints_current, descriptors_current = (
        orb.detectAndCompute(
            gray_current,
            None
        )
    )

    if (
        descriptors_reference is None
        or descriptors_current is None
    ):
        return current, False, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        descriptors_reference,
        descriptors_current
    )

    matches = sorted(
        matches,
        key=lambda match: match.distance
    )

    if len(matches) < 4:
        return current, False, 0

    selected_matches = matches[:50]

    points_reference = np.float32(
        [
            keypoints_reference[m.queryIdx].pt
            for m in selected_matches
        ]
    ).reshape(-1, 1, 2)

    points_current = np.float32(
        [
            keypoints_current[m.trainIdx].pt
            for m in selected_matches
        ]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        points_current,
        points_reference,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        return current, False, 0

    aligned = cv2.warpPerspective(
        current,
        homography,
        (
            reference.shape[1],
            reference.shape[0]
        )
    )

    inliers = (
        int(mask.sum())
        if mask is not None
        else 0
    )

    return aligned, True, inliers


def create_change_mask(reference, aligned):
    gray_reference = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    gray_aligned = cv2.cvtColor(
        aligned,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        gray_reference,
        gray_aligned
    )

    difference = cv2.GaussianBlur(
        difference,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def find_changed_region(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= MIN_CHANGE_AREA:
            x, y, width, height = cv2.boundingRect(
                contour
            )

            valid_regions.append(
                (
                    area,
                    x,
                    y,
                    width,
                    height
                )
            )

    if not valid_regions:
        return None

    valid_regions.sort(
        reverse=True
    )

    _, x, y, width, height = valid_regions[0]

    return x, y, width, height


def crop_region(image, region):
    x, y, width, height = region

    x1 = max(
        0,
        x - PADDING
    )

    y1 = max(
        0,
        y - PADDING
    )

    x2 = min(
        image.shape[1],
        x + width + PADDING
    )

    y2 = min(
        image.shape[0],
        y + height + PADDING
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop, (
        x1,
        y1,
        x2,
        y2
    )


def calculate_ai_error(model, crop):
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image = Image.fromarray(
        gray
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():
        reconstructed = model(
            tensor
        )

        error = torch.mean(
            (tensor - reconstructed) ** 2
        ).item()

    return error


def create_output_image(
    image,
    mask,
    region,
    confirmed
):
    output = image.copy()

    highlighted = np.zeros_like(
        output
    )

    highlighted[:, :, 2] = mask

    output = cv2.addWeighted(
        output,
        0.75,
        highlighted,
        0.35,
        0
    )

    if region is not None:
        x1, y1, x2, y2 = region

        if confirmed:
            rectangle_color = (
                0,
                0,
                255
            )
        else:
            rectangle_color = (
                0,
                255,
                255
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            rectangle_color,
            2
        )

    return output


def save_detection_results(results):
    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "change_ratio",
                "ai_error",
                "temporal_change",
                "ai_confirmed",
                "x",
                "y",
                "width",
                "height"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()
    threshold = load_threshold()

    session_1_images = sorted(
        SESSION_1_DIR.glob(
            "frame_*.jpg"
        )
    )

    session_2_images = sorted(
        SESSION_2_DIR.glob(
            "frame_*.jpg"
        )
    )

    pair_count = min(
        len(session_1_images),
        len(session_2_images)
    )

    print()
    print(
        "AquaSense Temporal + Localized AI Confirmation"
    )
    print(
        "-----------------------------------------------"
    )

    print(
        f"Session 1 images : {len(session_1_images)}"
    )

    print(
        f"Session 2 images : {len(session_2_images)}"
    )

    print(
        f"Pairs compared   : {pair_count}"
    )

    print(
        f"AI threshold     : {threshold:.6f}"
    )

    print()

    temporal_changes = 0
    confirmed_anomalies = 0

    detection_results = []

    for index in range(pair_count):
        reference = cv2.imread(
            str(
                session_1_images[index]
            )
        )

        current = cv2.imread(
            str(
                session_2_images[index]
            )
        )

        if (
            reference is None
            or current is None
        ):
            continue

        aligned, aligned_ok, inliers = align_frames(
            reference,
            current
        )

        if not aligned_ok:
            print(
                f"Pair {index + 1:02d} | "
                "ALIGNMENT FAILED"
            )

            continue

        mask = create_change_mask(
            reference,
            aligned
        )

        change_ratio = float(
            np.mean(mask > 0)
        )

        temporal_change = (
            change_ratio > CHANGE_THRESHOLD
        )

        region = None
        ai_error = 0.0
        ai_confirmed = False
        output = aligned.copy()

        x = 0
        y = 0
        width = 0
        height = 0

        if temporal_change:
            temporal_changes += 1

            region = find_changed_region(
                mask
            )

            if region is not None:
                crop, crop_coordinates = crop_region(
                    aligned,
                    region
                )

                ai_error = calculate_ai_error(
                    model,
                    crop
                )

                ai_confirmed = (
                    ai_error > threshold
                )

                output = create_output_image(
                    aligned,
                    mask,
                    crop_coordinates,
                    ai_confirmed
                )

                x1, y1, x2, y2 = (
                    crop_coordinates
                )

                x = x1
                y = y1
                width = x2 - x1
                height = y2 - y1

                if ai_confirmed:
                    confirmed_anomalies += 1
                    status = (
                        "NEW ANOMALY CONFIRMED"
                    )
                else:
                    status = (
                        "TEMPORAL CHANGE ONLY"
                    )

            else:
                status = (
                    "CHANGE REGION TOO SMALL"
                )

                output = create_output_image(
                    aligned,
                    mask,
                    None,
                    False
                )

        else:
            status = (
                "NO SIGNIFICANT CHANGE"
            )

        detection_results.append(
            {
                "frame": index + 1,
                "change_ratio": f"{change_ratio:.6f}",
                "ai_error": f"{ai_error:.6f}",
                "temporal_change": temporal_change,
                "ai_confirmed": ai_confirmed,
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        )

        output_path = (
            OUTPUT_DIR /
            f"comparison_{index + 1:04d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            output
        )

        print(
            f"Pair {index + 1:02d} | "
            f"ALIGNED | "
            f"Change: {change_ratio:.4f} | "
            f"AI error: {ai_error:.6f} | "
            f"{status}"
        )

    save_detection_results(
        detection_results
    )

    print()
    print(
        "Temporal + Localized AI analysis complete."
    )
    print(
        "---------------------------------------------"
    )

    print(
        f"Pairs processed        : {pair_count}"
    )

    print(
        f"Temporal changes       : {temporal_changes}"
    )

    print(
        f"Confirmed new anomalies: {confirmed_anomalies}"
    )

    print(
        f"Detection CSV          : {RESULTS_CSV}"
    )

    print(
        f"Output folder          : {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()