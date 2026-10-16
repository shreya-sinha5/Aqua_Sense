from pathlib import Path
import argparse
import time

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_DIR / "data" / "sessions"


def capture_session(session_name, frame_count=30, duration=5):
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Remove old frames so every session contains only the latest capture.
    for old_frame in session_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print()
    print("AquaSense Temporal Session Capture")
    print("-----------------------------------")
    print(f"Session name : {session_name}")
    print(f"Frames       : {frame_count}")
    print(f"Duration     : {duration} seconds")
    print()
    print("Press ENTER in this terminal to start capture.")
    input()

    print()
    print("Capturing...")
    print("Press Q in the camera window to stop early.")
    print()

    start_time = time.time()
    frames_saved = 0

    while frames_saved < frame_count:
        elapsed = time.time() - start_time

        if elapsed >= duration:
            break

        ret, frame = camera.read()

        if not ret:
            print("WARNING: Could not read frame from webcam.")
            continue

        frame_number = frames_saved + 1

        frame_path = (
            session_dir /
            f"frame_{frame_number:04d}.jpg"
        )

        cv2.imwrite(
            str(frame_path),
            frame
        )

        frames_saved += 1

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            "AquaSense DEMO - Webcam stand-in for sonar",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Session: {session_name} | Frame: {frames_saved}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AquaSense Session Capture",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Capture stopped by user.")
            break

        remaining_frames = frame_count - frames_saved
        remaining_time = duration - elapsed

        if remaining_frames > 0 and remaining_time > 0:
            delay = remaining_time / remaining_frames
            time.sleep(min(delay, 0.5))

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("Session capture complete.")
    print("------------------------")
    print(f"Frames saved : {frames_saved}")
    print(f"Location     : {session_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture AquaSense temporal webcam sessions."
    )

    parser.add_argument(
        "--session_name",
        required=True,
        help="Name of the session, for example session_1"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to capture"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Capture duration in seconds"
    )

    args = parser.parse_args()

    if args.frames <= 0 or args.duration <= 0:
        print("ERROR: Frames and duration must be greater than zero.")
        return

    if args.session_name == "session_2":
        print()
        print("IMPORTANT")
        print("---------")
        print(
            "Move or add an object in the scene to simulate "
            "new debris before capturing."
        )
        print()

    capture_session(
        args.session_name,
        args.frames,
        args.duration
    )


if __name__ == "__main__":
    main()