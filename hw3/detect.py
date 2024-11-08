from ultralyticsplus import YOLO, render_result
import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np

import numpy as np
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import math

# image_reid_person = pipeline(Tasks.image_reid_person, model='damo/cv_passvitb_image-reid-person_market')

def is_near_edge(bbox, frame_width, frame_height, edge_ratio):
    x1, y1, x2, y2 = bbox
    return x1 < frame_width * edge_ratio or x2 > frame_width * (1 - edge_ratio) or y1 < frame_height * edge_ratio or y2 > frame_height * (1 - edge_ratio)

def find_nearest_disappeared_id(bbox, disappeared_buffer, frame_width, frame_height):
    nearest_id = None
    min_distance = math.sqrt((frame_width / 5) ** 2 + (frame_height / 5) ** 2)  # Set a large distance as the initial value
    cb_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    for disappeared_id, (db_bbox, _) in disappeared_buffer.items():
        db_center = np.array([(db_bbox[0] + db_bbox[2]) / 2, (db_bbox[1] + db_bbox[3]) / 2])
        distance = np.linalg.norm(cb_center - db_center)
        print(f'distance: {distance}, min_distance: {min_distance}, disappeared_id: {disappeared_id}, dp_bbox: {db_bbox}')
        if distance < min_distance:
            min_distance = distance
            nearest_id = disappeared_id

    return nearest_id

def calculate_reid_score(image1, image2):
    

    result_1 = image_reid_person(image1)
    result_2 = image_reid_person(image2)

    feat_1 = np.array(result_1[OutputKeys.IMG_EMBEDDING][0])
    feat_2 = np.array(result_2[OutputKeys.IMG_EMBEDDING][0])

    feat_norm_1 = feat_1 / np.linalg.norm(feat_1)
    feat_norm_2 = feat_2 / np.linalg.norm(feat_2)

    score = np.dot(feat_norm_1, feat_norm_2)
    return score

def hungarian_algorithm(previous_boxes, current_boxes, id_to_image_buffer, frame_width, frame_height, disappeared_buffer):
    if len(previous_boxes) == 0 or len(current_boxes) == 0:
        return []

    cost_matrix = np.zeros((len(previous_boxes), len(current_boxes)), dtype=np.float32)
    for i, pb in enumerate(previous_boxes):
        pb_image = id_to_image_buffer.get(previous_ids[i])  # Get the image from buffer
        for j, cb in enumerate(current_boxes):
            cb_image = extract_bbox_image(frame, cb)  # Extract current image
            iou_cost = 1 - iou(pb, cb)
            # if pb_image is not None:
            #     reid_cost = 1 - calculate_reid_score(pb_image, cb_image)  # Calculate ReID similarity
            # else:
            #     reid_cost = 1  # Max cost if no image is available
            cost_matrix[i, j] = iou_cost   # Combine costs

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(r, c, False) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 0.8]

    # find col_ind not in matches
    new_col_ind = []
    for c in col_ind:
        if c in [c for _, c, _ in matches]:
            new_col_ind.append(c)

    print(f'len(current_boxes): {len(current_boxes)}')
    print(f'Before matches: {matches}')

    # Check for new bounding boxes not near the edge
    for current_idx, cb in enumerate(current_boxes):
        if current_idx not in new_col_ind and not is_near_edge(cb, frame_width, frame_height, 0.01):
            # Find the nearest disappeared ID
            nearest_id = find_nearest_disappeared_id(cb, disappeared_buffer, frame_width, frame_height)
            if nearest_id is not None:
                matches.append((nearest_id, current_idx, True))
                del disappeared_buffer[nearest_id]

    print(f'After matches: {matches}')

    return matches

def iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IOU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def extract_bbox_image(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]

def update_disappeared_buffer(disappeared_buffer, previous_ids, previous_boxes, max_disappear_time, current_unique_ids, id_visibility_duration):
    # Increment the frame count for existing IDs in the buffer
    for id in list(disappeared_buffer.keys()):
        disappeared_buffer[id] = (disappeared_buffer[id][0], disappeared_buffer[id][1] + 1)
        if disappeared_buffer[id][1] > max_disappear_time:
            del disappeared_buffer[id]  # Remove if exceeded the max disappear time

    # Add newly disappeared IDs to the buffer
    for idx, id in enumerate(previous_ids):
        if id not in current_unique_ids and id in id_visibility_duration:  # current_unique_ids should be the IDs found in the current frame
            disappeared_buffer[id] = (previous_boxes[idx], 0)  # Add with frame count 0

    return disappeared_buffer

def remove_short_lived_ids(id_visibility_duration, threshold, current_unique_ids, next_id):
    for id, duration in list(id_visibility_duration.items()):
        if duration <= threshold and id not in current_unique_ids:
            # ID has disappeared and was visible for a short duration
            del id_visibility_duration[id]  # Remove from visibility duration tracking
            next_id -= 1  # Decrement the next_id counter

    return id_visibility_duration, next_id

def generate_unique_color(id, max_ids=20):
    # Scale the hue by the ID to get a wide range of colors
    hue = int((float(id) / max_ids) * 180)
    color_hsv = np.uint8([[[hue, 255, 255]]])  # Saturation and Value are set to max
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)

# load model
model = YOLO('yolov8x.pt')

# set model parameters
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 100  # maximum number of detections per image
model.overrides['classes'] = [0]

# Initialize variables
previous_boxes = []
current_boxes = []
previous_ids = []

next_id = 0

id_to_image_buffer = {}
disappeared_buffer = {}
id_visibility_duration = {}
id_to_color = {}

# Open the video file
video_path = "easy_9.mp4"
cap = cv2.VideoCapture(video_path)

# save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'{video_path.split(".")[0]}_output.mp4', fourcc, 30.0, (1280, 720))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)
    current_boxes = results[0].boxes.xyxy.cpu().numpy()  # Boxes object for bbox outputs

    # Reset previous_ids for the current frame
    current_ids = [-1] * len(current_boxes)

    # Hungarian algorithm to match boxes
    matches = hungarian_algorithm(previous_boxes, current_boxes, id_to_image_buffer, frame.shape[1], frame.shape[0], disappeared_buffer)

    # Update tracked objects with matched IDs
    for (prev_idx, current_idx, use_prev) in matches:
        if use_prev:
            current_ids[current_idx] = prev_idx
            # next_id = max(next_id, previous_ids[prev_idx] + 1)
        else:
            current_ids[current_idx] = previous_ids[prev_idx]
        
    id_visibility_duration, next_id = remove_short_lived_ids(id_visibility_duration, 45, current_ids, next_id)

    # Assign new IDs to unmatched objects
    print(f'previous_ids: {previous_ids}')
    print(f'current_ids: {current_ids}')
    for i, id in enumerate(current_ids):
        if id == -1:
            current_ids[i] = next_id
            next_id += 1

    for box, id in zip(current_boxes, current_ids):
        bbox_image = extract_bbox_image(frame, box)
        id_to_image_buffer[id] = bbox_image

    # Update disappeared buffer
    disappeared_buffer = update_disappeared_buffer(disappeared_buffer, previous_ids, previous_boxes, 900, current_ids, id_visibility_duration)


    print(f'Disappeared buffer: {list(disappeared_buffer.keys())}')

    for id in current_ids:
        if id in id_visibility_duration:
            id_visibility_duration[id] += 1
        else:
            id_visibility_duration[id] = 1

    print(f'ID visibility duration: {id_visibility_duration}')

    # After assigning IDs
    for id in current_ids:
        if id not in id_to_color:
            id_to_color[id] = generate_unique_color(id)

    # Draw bounding boxes and IDs
    for box, id in zip(current_boxes, current_ids):
        color = id_to_color[id]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, str(id), (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", frame)

    # save video
    out.write(frame)

    # Update for the next iteration
    previous_ids = current_ids.copy()
    previous_boxes = current_boxes.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Output the count result
print(f'-' * 30)
print(f'Count : {next_id}')  # next_id now reflects the total count
print(f'-' * 30)

cap.release()
cv2.destroyAllWindows()