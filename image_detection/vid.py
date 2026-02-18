import cv2
from ultralytics import YOLO
import os
import time

# ---------------- CONFIG ----------------
VIDEO_PATH = "test_vid1.mp4"              # your test video
MODEL_PATH = "accident.pt"               # your trained model
CONF_THRESHOLD = 0.4                 # lower = fewer misses
IMG_SIZE = 960                       # higher = better small object detection
SAVE_FOLDER = "accident_frames"
COOLDOWN_SECONDS = 2                 # prevents duplicate saving
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Print class names once (debug safety)
print("Model classes:", model.names)

# Create save folder
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video file.")
    exit()

last_saved_time = 0

print("Running accident detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)

    accident_detected = False

    # Check detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        # Flexible check (handles 'accidents')
        if "accident" in label.lower():
            accident_detected = True

    # Save frame if accident found
    current_time = time.time()

    if accident_detected and (current_time - last_saved_time > COOLDOWN_SECONDS):
        filename = f"{SAVE_FOLDER}/accident_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        print("Accident detected → Frame saved:", filename)
        last_saved_time = current_time

    # Let YOLO draw boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Accident Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection finished.")
