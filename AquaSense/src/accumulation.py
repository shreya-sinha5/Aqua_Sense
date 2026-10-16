from pathlib import Path
from collections import defaultdict
import csv


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "temporal_ai"
    / "temporal_detections.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "accumulation"
)

OUTPUT_CSV = OUTPUT_DIR / "priority_plan.csv"

GRID_ROWS = 3
GRID_COLS = 3

HIGH_PRIORITY_SCORE = 70
MEDIUM_PRIORITY_SCORE = 40


def get_grid_cell(x, y, width, height):
    column = min(
        int((x / width) * GRID_COLS),
        GRID_COLS - 1
    )

    row = min(
        int((y / height) * GRID_ROWS),
        GRID_ROWS - 1
    )

    return f"G{row + 1}{column + 1}"


def calculate_priority(detection_count, change_strength):
    persistence_score = min(
        detection_count * 15,
        60
    )

    change_score = min(
        change_strength * 40,
        40
    )

    priority_score = (
        persistence_score
        + change_score
    )

    if priority_score >= HIGH_PRIORITY_SCORE:
        priority = "HIGH"
    elif priority_score >= MEDIUM_PRIORITY_SCORE:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return priority_score, priority


def main():
    if not INPUT_CSV.exists():
        print()
        print("ERROR: Temporal detection CSV was not found.")
        print(f"Expected file: {INPUT_CSV}")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detections_by_cell = defaultdict(list)

    with open(
        INPUT_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["temporal_change"].lower() != "true":
                continue

            x = float(row["x"])
            y = float(row["y"])
            width = float(row["width"])
            height = float(row["height"])

            change_ratio = float(
                row["change_ratio"]
            )

            cell = get_grid_cell(
                x,
                y,
                640,
                480
            )

            detections_by_cell[cell].append(
                {
                    "frame": int(row["frame"]),
                    "change": change_ratio
                }
            )

    results = []

    for cell, detections in detections_by_cell.items():
        detection_count = len(detections)

        average_change = sum(
            detection["change"]
            for detection in detections
        ) / detection_count

        priority_score, priority = calculate_priority(
            detection_count,
            average_change
        )

        results.append(
            {
                "grid_cell": cell,
                "detections": detection_count,
                "average_change": average_change,
                "priority_score": priority_score,
                "priority": priority
            }
        )

    results.sort(
        key=lambda item: item["priority_score"],
        reverse=True
    )

    print()
    print("AquaSense Accumulation + Priority Planner")
    print("-----------------------------------------")

    if not results:
        print()
        print("No temporal detections available.")
        print("Run temporal_ai.py first.")
        return

    print()
    print("Accumulation Analysis")
    print("---------------------")

    for result in results:
        print(
            f"{result['grid_cell']} | "
            f"Detections: {result['detections']:02d} | "
            f"Avg change: {result['average_change']:.4f} | "
            f"Priority score: {result['priority_score']:.2f} | "
            f"{result['priority']}"
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "grid_cell",
                "detections",
                "average_change",
                "priority_score",
                "priority"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("Mission Priority Queue")
    print("----------------------")

    for index, result in enumerate(
        results,
        start=1
    ):
        print(
            f"{index}. "
            f"{result['grid_cell']} -> "
            f"{result['priority']} priority"
        )

    print()
    print(
        "Accumulation + priority planning complete."
    )
    print("------------------------------------------")
    print(
        f"Priority plan saved : {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()