from ultralytics import YOLO

model = YOLO("accidents.pt")

results = model("image.jpg")

results[0].show()