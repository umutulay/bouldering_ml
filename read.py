from ultralytics import YOLO
import torch
import multiprocessing

def main():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO("yolov8n.pt")  # or yolov8s.pt
    model.train(data="climbing_dataset/data.yaml", epochs=50, imgsz=416, batch=4)

    # Run inference on an image
    results = model("bouldering_route.png")
    results[0].show()      # show detections
    results[0].save()      # save annotated image

    # Optional: Extract bounding box data
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        print("Hold at:", x1.item(), y1.item(), x2.item(), y2.item())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
