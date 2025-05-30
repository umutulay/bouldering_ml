from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

results = model("route1.jpg", save=True, conf=0.25)
