from ultralytics import YOLO

model = YOLO("best.pt")

results = model("image.jpg")

results[0].show()