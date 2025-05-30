from ultralytics import YOLO
import cv2
import math

# Load YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")

# Input image path
img_path = "route_examples/climbing_route_new.png"

# Run detection
results = model(img_path, conf=0.5)

# Load original image
img = cv2.imread(img_path)

# Check if image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

# Extract detection results
boxes = results[0].boxes.xyxy.cpu().numpy()
confidences = results[0].boxes.conf.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()

print(f"Detected {len(boxes)} holds")

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
def find_route(holds, max_reach=150, start_hold=None):
    if not holds:
        return []

    if start_hold:
        current = start_hold
        route = [start_hold]
    else:
        current = holds[0]
        route = [current]

    used = set()
    used.add(current)

    while True:
        candidates = [
            h for h in holds
            if h != current and h not in used and distance(current, h) <= max_reach
        ]
        if not candidates:
            break
        next_hold = min(candidates, key=lambda h: distance(current, h))
        route.append(next_hold)
        used.add(next_hold)
        current = next_hold

    return route


# Initialize font variables BEFORE they're used
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2

selected_point = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point.append((x, y))
        print(f"Selected start point: ({x}, {y})")

# Show image and wait for click
preview_img = img.copy()
for hold in holds:
    cv2.circle(preview_img, (int(hold[0]), int(hold[1])), 6, (255, 0, 0), 2)

cv2.imshow("Click START Hold", preview_img)
cv2.setMouseCallback("Click START Hold", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------- Use selected point to define route --------
if selected_point:
    start_click = selected_point[0]

    # Show clicked point
    cv2.circle(img, start_click, 10, (0, 255, 255), 3)
    cv2.putText(img, "CLICKED", (start_click[0] + 10, start_click[1]), font, font_scale, (0, 255, 255), font_thickness)

    # Find the closest hold to the clicked point
    distances = [distance(h, start_click) for h in holds]
    min_dist = min(distances)
    start_hold_index = distances.index(min_dist)
    start_hold = holds[start_hold_index]

    print(f"Clicked: {start_click}")
    print(f"Selected START hold: {start_hold}, distance: {min_dist:.2f}")

    # Mark selected hold
    cv2.circle(img, (int(start_hold[0]), int(start_hold[1])), 12, (0, 165, 255), 2)
    cv2.putText(img, "SELECTED HOLD", (int(start_hold[0] + 10), int(start_hold[1])), font, font_scale, (0, 165, 255), font_thickness)

    # Route calculation - Fix the route calculation logic
    # Create a new holds list starting with the selected hold
    remaining_holds = [h for h in holds if h != start_hold]
    # Find route starting from the selected hold
    print("\nAll detected holds and distances from start:")
    for h in holds:
        if h == start_hold:
            continue
        print(f"Hold: {h}, y_diff: {start_hold[1] - h[1]:.2f}, distance: {distance(start_hold, h):.2f}")

    route_from_start = find_route(holds, max_reach=800, start_hold=start_hold)
    route = route_from_start

else:
    route = find_route(holds, max_reach=600)

# Draw all holds as blue circles
for hold in holds:
    cv2.circle(img, (int(hold[0]), int(hold[1])), 6, (255, 0, 0), 2)

# Draw the predicted route
for i in range(len(route) - 1):
    pt1 = tuple(map(int, route[i]))
    pt2 = tuple(map(int, route[i + 1]))
    cv2.line(img, pt1, pt2, (0, 255, 0), 3)
    cv2.circle(img, pt1, 8, (0, 255, 0), -1)

# Mark the last hold as well
if route:
    cv2.circle(img, tuple(map(int, route[-1])), 8, (0, 255, 0), -1)

# Label each hold in the route with its step number
for i, pt in enumerate(route):
    cv2.putText(img, str(i + 1), (int(pt[0] + 5), int(pt[1] - 5)), font, 0.6, (255, 255, 255), 2)


# Mark START and TOP
if route:
    start_pt = tuple(map(int, route[0]))
    top_pt = tuple(map(int, route[-1]))

    cv2.circle(img, start_pt, 8, (0, 255, 0), -1)  # START = Green
    cv2.putText(img, "START", (start_pt[0] + 10, start_pt[1] - 10), font, font_scale, (0, 255, 0), font_thickness)

    cv2.circle(img, top_pt, 8, (0, 0, 255), -1)    # TOP = Red
    cv2.putText(img, "TOP", (top_pt[0] + 10, top_pt[1] - 10), font, font_scale, (0, 0, 255), font_thickness)

# Display the result
cv2.imshow("Predicted Route", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("predicted_route_labeled.png", img)
print("Labeled route saved as 'predicted_route_labeled.png'")
print(f"Route found with {len(route)} holds")