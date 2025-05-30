import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import random

# --- Load YOLO model ---
model = YOLO("runs/detect/train2/weights/best.pt")  # or your custom-trained weights

# --- Load image ---
img_path = "route1.jpg"
image = cv2.imread(img_path)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Run detection ---
results = model(img_path)
boxes = results[0].boxes
colors = []

# --- Extract dominant color from each hold ---
def get_dominant_color(img_crop, k=1):
    pixels = img_crop.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]  # RGB

# Extract cropped box colors
cropped_boxes = []
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    hold_crop = img_rgb[y1:y2, x1:x2]
    if hold_crop.size == 0: continue
    dom_color = get_dominant_color(hold_crop)
    colors.append(dom_color)
    cropped_boxes.append((x1, y1, x2, y2))

# --- Cluster holds by color ---
num_routes = 4  # how many route colors to classify
kmeans = KMeans(n_clusters=num_routes, n_init=10)
labels = kmeans.fit_predict(colors)

# --- Visualize results ---
route_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_routes)]
for i, (x1, y1, x2, y2) in enumerate(cropped_boxes):
    color = route_colors[labels[i]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"Route {labels[i]}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# --- Show and save the result ---
cv2.imwrite("grouped_routes.jpg", image)
print("Saved as grouped_routes.jpg")
