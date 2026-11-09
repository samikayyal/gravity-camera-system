from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = Path("data/Camera 1.mp4")
MODEL_PATH = "yolo26n.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # results = model.track(
    #     frame,
    #     persist=True,
    #     tracker="bytetrack.yaml",
    #     classes=[0],  # Person class only
    #     conf=0.25,
    #     verbose=False,
    # )
    results = model.predict(
        frame,
        imgsz=1280,
        conf=0.10,
        classes=[0],  # Person class only
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()

    player_count = len(result.boxes)

    cv2.putText(
        annotated,
        f"Players: {player_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    # Resize display while retaining original
    # resolution for inference.
    h, w = annotated.shape[:2]
    scale = min(1.0, 1280 / w, 720 / h)
    display = cv2.resize(
        annotated,
        (int(w * scale), int(h * scale)),
    )

    cv2.imshow("Gravity - Player Tracking", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Save original frame for zone annotation
    if key == ord("s"):
        cv2.imwrite("trampoline_frame.png", frame)
        print("Saved trampoline_frame.png")

cap.release()
cv2.destroyAllWindows()
