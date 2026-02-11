from ultralytics import YOLO

model=YOLO("yolov8n.pt")

def detect_yolo(frames):

    results=model.predict(source=frames, conf=0.25,iou=0.5,imgsz=1280,device="0",verbose=False)

    return results[0]