from ultralytics import YOLO

def export_yolo_to_onnx():
    model = YOLO("yolov8n.pt")
    
    model.export(
        format="onnx",
        imgsz=640,        # input size
        dynamic=False,    # fixed input shape for TensorRT later
        simplify=True,    # simplify graph — removes redundant ops
        opset=12          # ONNX opset version — 12 is widely supported
    )
    print("Exported to yolov8n.onnx")

if __name__ == "__main__":
    export_yolo_to_onnx()