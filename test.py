from ultralytics import YOLO

model = YOLO("best.pt")

results = model("dog.jpeg")

results[0].show()