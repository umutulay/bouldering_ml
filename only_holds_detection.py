from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

results = model("route_examples/climbing_route_new.png", save=True, conf=0.5) # Change the confidence threshold as needed
