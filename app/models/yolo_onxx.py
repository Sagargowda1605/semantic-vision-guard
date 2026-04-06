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