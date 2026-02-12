from ultralytics import YOLO

model = YOLO("yolo26s-cls.pt")

results = model.train(data = "custom_dataset", epochs = 10, imgsz = 640)