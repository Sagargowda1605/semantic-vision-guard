from ultralytics import YOLO

model=YOLO("yolov8n.pt")

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 15: "cat", 16: "dog",
    17: "horse", 24: "backpack", 26: "handbag"
}

def detect_yolo(frames):

    results=model.predict(source=frames, conf=0.25,iou=0.5,imgsz=1280,device="0",verbose=False)

    return results[0]
#the yolo returns the results in the form of a list of objects, each containing 
# the bounding boxes, class ids, and confidence scores for the detected objects in the input frames.
#the results[0] is used to access the first element of the list, which contains the detection results for the input frames. 
# This is because the model.predict() method can return multiple results if multiple frames are processed at once, but in this case,
#  we are only interested in the first result.