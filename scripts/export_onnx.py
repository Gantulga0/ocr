"""Export detector + recognizer to ONNX (for TensorRT / edge deployment in the parking system)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mn_alpr.crnn import CRNN, IMG_H, IMG_W
from ultralytics import YOLO
os.makedirs("models/onnx", exist_ok=True)
YOLO("models/plate_det_yolo11n.pt").export(format="onnx", imgsz=640, half=False, dynamic=False, simplify=True)
os.replace("models/plate_det_yolo11n.onnx", "models/onnx/plate_det_yolo11n.onnx")
m = CRNN().eval(); m.load_state_dict(torch.load("models/plate_rec_crnn.pt", map_location="cpu"))
torch.onnx.export(m, torch.zeros(1, 3, IMG_H, IMG_W), "models/onnx/plate_rec_crnn.onnx", input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
print("exported to models/onnx/")
