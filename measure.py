from ultralytics import YOLO
import cv2
import numpy as np
import os

# ======================
# KONFIGURASI
# ======================
MODEL_PATH = "runs/segment/train/weights/best.pt"
IMAGE_DIR = "test_images"
IMAGE_LIST = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]

CAP_CLASS = 0
BODY_CLASS = q

TOTAL_HEIGHT_CM = 23.0
MIN_BODY_HEIGHT_CM = 18.0

# ======================
# LOAD MODEL SEGMENT
# ======================
model = YOLO(MODEL_PATH)

print("\n=== SEGMENTATION MEASUREMENT START ===\n")

for img_name in IMAGE_LIST:
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"{img_name}: IMAGE NOT FOUND ❌\n")
        continue

    results = model(image, conf=0.25, verbose=False)

    if results[0].masks is None:
        print(f"{img_name}: NO SEGMENTATION ❌\n")
        continue

    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    body_heights = []
    cap_heights = []

    for mask, cls in zip(masks, classes):
        ys, _ = np.where(mask == 1)
        if len(ys) == 0:
            continue

        height_px = ys.max() - ys.min()

        if cls == BODY_CLASS:
            body_heights.append(height_px)
        elif cls == CAP_CLASS:
            cap_heights.append(height_px)

    if not body_heights:
        print(f"{img_name}: BODY NOT FOUND ❌\n")
        continue

    body_px = max(body_heights)
    cap_px = max(cap_heights) if cap_heights else 0
    total_px = body_px + cap_px

    scale = TOTAL_HEIGHT_CM / total_px
    body_cm = body_px * scale
    cap_cm = cap_px * scale

    print(f"IMAGE: {img_name}")
    print(f"Body Height : {body_cm:.2f} cm")
    print(f"Cap Height  : {cap_cm:.2f} cm")

    if body_cm < MIN_BODY_HEIGHT_CM:
        print("STATUS: FAIL ❌ (Tiang terlalu pendek)\n")
    else:
        print("STATUS: PASS ✅\n")

print("=== DONE ===")

