import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2

def IOU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2
    
    # Calculate intersection coordinates
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    
    # Compute intersection area
    inter_width, inter_height = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Compute both bounding box areas
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x2_p - x1_p) * (y2_p - y1_p)
    
    # Compute union area
    union_area = area_bbox1 + area_bbox2 - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Compute IoU
    return inter_area / union_area

class Tracker:
    def __init__(self):
        self.trackers = []
        self.disappeared_trackers = []
        self.max_age = 10
        self.min_hits = 3
        self.next_id = 1
    
    def update(self, current_bboxs, features, iou_weight=0.5, reid_weight=0.5):
        n_detections = len(current_bboxs)
        n_tracks = len(self.trackers)
        if n_tracks == 0 or n_detections == 0:
            unmatched_detections = set(range(n_detections))
            unmatched_trackers = set(range(n_tracks))
            
            # Initialize new trackers for each unmatched detection if there are detections
            for d in unmatched_detections:
                self.trackers.append(Track(current_bboxs[d], features[d], self.next_id))
                self.next_id += 1

            # For unmatched trackers, you can also handle any missed count if needed
            for t in unmatched_trackers:
                self.trackers[t].age += 1
                if self.trackers[t].age > self.max_age:
                    self.trackers.pop(t)

            return  


        # Assign detections to existing trackers using Hungarian algorithm
        cost_matrix = np.zeros((len(self.trackers), len(current_bboxs)), dtype=np.float32)
        iou_cost_matrix = np.zeros((len(self.trackers), len(current_bboxs)), dtype=np.float32)
        reid_cost_matrix = np.zeros((len(self.trackers), len(current_bboxs)), dtype=np.float32)
        for t, tracker in enumerate(self.trackers):
            for d, bbox in enumerate(current_bboxs):
                reid_loss = np.linalg.norm(tracker.feature - features[d])  # ReID feature distance
                iou_loss = 1 - IOU(tracker.bbox, bbox)
                iou_cost_matrix[t, d] = iou_loss
                reid_cost_matrix[t, d] = reid_loss
        
        # Normalize the costs
        iou_cost_matrix /= iou_cost_matrix.max() if iou_cost_matrix.max() > 0 else 1
        reid_cost_matrix /= reid_cost_matrix.max() if reid_cost_matrix.max() > 0 else 1

        cost_matrix = (iou_weight * iou_cost_matrix) + (reid_weight * reid_cost_matrix)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched, disappered_trackers, new_detections = [], [], []

        # Handle matches, unmatches, and create new trackers
        for t, tracker in enumerate(self.trackers):
            if t in row_ind:
                matched.append((t, col_ind[list(row_ind).index(t)]))
            else:
                disappered_trackers.append(t)

        for d, detection in enumerate(current_bboxs):
            if d not in col_ind:
                new_detections.append(d)

        # Update matched trackers
        for t, d in matched:
            self.trackers[t].update(current_bboxs[d], features[d])

        # Handle disappered_trackers
        for t in disappered_trackers:
            self.trackers[t].age += 1
            if self.trackers[t].age > self.max_age:
                self.trackers.pop(t)

        # Create new trackers for unmatched detections
        for d in new_detections:
            self.trackers.append(Track(current_bboxs[d], features[d], self.next_id))
            self.next_id += 1



    

class Track:
    def __init__(self, bbox, feature, track_id):
        self.bbox = bbox
        self.feature = feature
        self.track_id = track_id
        self.kalman_filter = KalmanFilter(dim_x=7, dim_z=4)
        self.kalman_filter.x[:4] = np.reshape(bbox, (4, 1))  # Initialize Kalman state with bbox
        self.age = 0
        self.color = self.get_unique_color()
    
    def update(self, bbox, feature):
        self.bbox = bbox
        self.feature = feature
        self.age = 0
        # Update Kalman filter state here

    def get_unique_color(self, max_ids=15):
        # Scale the hue by the ID to get a wide range of colors
        hue = int((float(self.track_id) / max_ids) * 180)
        color_hsv = np.uint8([[[hue, 255, 255]]])  # Saturation and Value are set to max
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)
        
