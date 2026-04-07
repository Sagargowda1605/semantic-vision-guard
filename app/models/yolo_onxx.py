import onnxruntime as rt
import numpy as np
import cv2 as cv

available = rt.get_available_providers()
print(f"Available providers: {available}")

# Load ONNX model once
session = rt.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
print(f"ONNX model loaded. Input: {input_name}")

def preprocess(frame):
    img = cv.resize(frame, (640, 640))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)     # add batch dim
    return img

def detect_yolo_onnx(frame):
    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0]  # shape (1, 84, 8400)

def postprocess(outputs, orig_frame, conf_threshold=0.25, iou_threshold=0.5):
    # outputs shape: (1, 84, 8400)
    predictions = outputs[0].transpose(1, 0)  # → (8400, 84)
    
    orig_h, orig_w = orig_frame.shape[:2]
    
    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        # Extract class scores (last 80)
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < conf_threshold:
            continue

        # Extract box coordinates (first 4) — x_center, y_center, w, h
        x_center, y_center, w, h = pred[:4]

        # Convert from 640x640 space to original frame size
        x_center = x_center * orig_w / 640
        y_center = y_center * orig_h / 640
        w = w * orig_w / 640
        h = h * orig_h / 640

        # Convert center format to corner format (x1,y1,x2,y2)
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))
        class_ids.append(int(class_id))

    # Apply NMS
    indices = cv.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, iou_threshold
    )

    final_boxes = []
    final_confs = []
    final_cls = []

    for i in indices:
        final_boxes.append(boxes[i])
        final_confs.append(confidences[i])
        final_cls.append(class_ids[i])

    return final_boxes, final_confs, final_cls