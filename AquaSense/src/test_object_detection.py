import cv2

from src.infer import AquaSenseInference


def main():
    print("Loading AquaSense AI...")
    print()

    inference = AquaSenseInference()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Camera started.")
    print()
    print("Controls:")
    print("  SPACE = capture and analyze")
    print("  Q     = quit")
    print()

    last_result = None

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        display = frame.copy()

        # Live pen detection

        detection = inference.detect_green_pen(
            frame
        )

        if detection is not None:

            display = inference.draw_detection(
                display,
                detection
            )

            cv2.putText(
                display,
                "AQUASENSE DEMO OBJECT",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "Show the green pen",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Last analysis result

        if last_result is not None:

            cv2.putText(
                display,
                f"Object: {last_result['label']}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Confidence: "
                f"{last_result['confidence'] * 100:.1f}%",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Anomaly: "
                f"{last_result['anomaly']:.5f}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            display,
            "SPACE = Analyze | Q = Quit",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AquaSense Object Detection",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # Analyze captured frame

        if key == 32:

            print()
            print("=" * 45)
            print("CAPTURED FRAME")
            print("=" * 45)
            print("Running AquaSense analysis...")
            print("Please wait...")
            print()

            results = inference.analyze(
                frame
            )

            objects = results["objects"]

            if objects:

                best = objects[0]

                last_result = {
                    "label": best["label"],
                    "confidence": best["confidence"],
                    "anomaly": results[
                        "anomaly_score"
                    ],
                }

                print(
                    f"Object: {best['label']}"
                )

                print(
                    f"Confidence: "
                    f"{best['confidence'] * 100:.2f}%"
                )

                print(
                    f"Anomaly score: "
                    f"{results['anomaly_score']:.6f}"
                )

                if results["detected"]:

                    print()
                    print(
                        "GREEN PEN DETECTOR: CONFIRMED"
                    )

            else:

                print(
                    "No object detected."
                )

            print()
            print("Analysis complete.")

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()