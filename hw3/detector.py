import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolo11x.pt'):
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = 0.25  # NMS confidence threshold
        self.model.overrides['iou'] = 0.4  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 100  # maximum number of detections per image
        self.model.overrides['classes'] = [0]
    
    def detect(self, frame):
        results = self.model(frame)
        return results[0].boxes.xyxy.cpu().numpy()  # Return bounding boxes in xyxy format
