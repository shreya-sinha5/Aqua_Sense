from pathlib import Path
from ultralytics import YOLOWorld
import cv2

# Load pre-trained zero-shot YOLO-World model
model = YOLOWorld("yolov8s-world.pt")

# Define the prompts for sonar objects
model.set_classes(["shipwreck", "aircraft", "fish", "underwater debris"])

base_dir = Path(__file__).resolve().parent.parent
input_dir = base_dir / "data" / "raw" / "debris"
output_img_dir = base_dir / "data" / "yolo_dataset" / "train" / "images"
output_lbl_dir = base_dir / "data" / "yolo_dataset" / "train" / "labels"

output_img_dir.mkdir(parents=True, exist_ok=True)
output_lbl_dir.mkdir(parents=True, exist_ok=True)

images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
print(f"[INFO] Auto-labeling {len(images)} images using YOLO-World...")

for img_path in images:
    # Run quick zero-shot detection
    results = model.predict(str(img_path), conf=0.15, verbose=False)

    # Save bounding boxes in YOLO format (.txt)
    txt_path = output_lbl_dir / f"{img_path.stem}.txt"
    with open(txt_path, "w") as f:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            # Normalized x_center, y_center, width, height
            x, y, w, h = box.xywhn[0].tolist()
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # Copy image to dataset folder
    cv2.imwrite(str(output_img_dir / img_path.name), cv2.imread(str(img_path)))

print("[SUCCESS] Fast auto-labeling finished!")