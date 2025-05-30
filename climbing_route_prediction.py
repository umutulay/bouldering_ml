from ultralytics import YOLO
import cv2
import math
import numpy as np

# Load YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")

# Input image path
img_path = "climbing_route_new.png"

# Run detection
results = model(img_path, conf=0.5)

# Load original image
img = cv2.imread(img_path)

# Extract detection results
boxes = results[0].boxes.xyxy.cpu().numpy()
confidences = results[0].boxes.conf.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()

# Get center points of holds
holds = []
for box in boxes:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    holds.append((center_x, center_y))

# Distance function
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Route prediction
def find_route(holds, max_reach=150):
    if not holds:
        return []
    
    holds_sorted = sorted(holds, key=lambda h: h[1], reverse=True)  # from bottom to top
    route = [holds_sorted[0]]
    current = holds_sorted[0]

    while True:
        candidates = [
            h for h in holds
            if h[1] < current[1] and distance(current, h) <= max_reach
        ]
        if not candidates:
            break
        next_hold = min(candidates, key=lambda h: distance(current, h))
        route.append(next_hold)
        current = next_hold

    return route

# Predict route
route = find_route(holds, max_reach=150)

# Draw all holds as blue circles
for hold in holds:
    cv2.circle(img, (int(hold[0]), int(hold[1])), 6, (255, 0, 0), 2)

# Draw the predicted route
for i in range(len(route) - 1):
    pt1 = tuple(map(int, route[i]))
    pt2 = tuple(map(int, route[i + 1]))
    cv2.line(img, pt1, pt2, (0, 255, 0), 3)
    cv2.circle(img, pt1, 8, (0, 255, 0), -1)

# Mark START and TOP
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2

if route:
    start_pt = tuple(map(int, route[0]))
    top_pt = tuple(map(int, route[-1]))

    cv2.circle(img, start_pt, 8, (0, 255, 0), -1)  # START = Green
    cv2.putText(img, "START", (start_pt[0] + 10, start_pt[1] - 10), font, font_scale, (0, 255, 0), font_thickness)

    cv2.circle(img, top_pt, 8, (0, 0, 255), -1)    # TOP = Red
    cv2.putText(img, "TOP", (top_pt[0] + 10, top_pt[1] - 10), font, font_scale, (0, 0, 255), font_thickness)

# Save result
cv2.imwrite("predicted_route_labeled.png", img)
print("Labeled route saved as 'predicted_route_labeled.png'")
