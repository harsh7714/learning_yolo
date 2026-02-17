import cv2
from ultralytics import YOLO
import os
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "fire_smoke.pt"     # your trained fire/smoke model
CONF_THRESHOLD = 0.4
IMG_SIZE = 960
SAVE_FOLDER = "fire_smoke_frames"
COOLDOWN_SECONDS = 2
CAMERA_INDEX = 1   # 0 = default USB webcam (change to 1 if multiple cameras)
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

print("Model classes:", model.names)

# Create save folder
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Open USB webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_saved_time = 0

print("Running Fire & Smoke detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO inference
    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)

    fire_or_smoke_detected = False
    detected_label = ""

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls].lower()

        if "fire" in label or "smoke" in label:
            fire_or_smoke_detected = True
            detected_label = label

    current_time = time.time()

    if fire_or_smoke_detected and (current_time - last_saved_time > COOLDOWN_SECONDS):
        filename = f"{SAVE_FOLDER}/{detected_label}_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        print(f"{detected_label.upper()} detected → Frame saved:", filename)
        last_saved_time = current_time

    # Let YOLO draw boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Fire & Smoke Detection", annotated_frame)

    # Slight delay to make display smoother
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection finished.")
