from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

results = model("trial.png", save=True, conf=0.25)
