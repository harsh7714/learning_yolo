from ultralytics import YOLO

model = YOLO("fire_smoke.pt")

print("Class Names:", model.names)
print("Number of Classes:", len(model.names))