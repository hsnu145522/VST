import os
import json
from collections import defaultdict

HEIGHT = 1080
WIDTH = 1920

def Convert2coco(label_dir, output_path):
    coco_data = defaultdict(list)
    image_id = 0
    annotation_id = 0

    # Define categories
    coco_data["categories"] = [{"id": 0, "name": "car"}] 
    labels = os.listdir(label_dir)
    labels.sort()
    for label_file in labels: 
        if label_file.endswith('.txt'):
            # Extract image info
            image_info = {
                "file_name": label_file.replace('.txt', '.jpg'), 
                "height": HEIGHT,
                "width": WIDTH,
                "id": image_id
            }
            coco_data["images"].append(image_info)

            # Read annotations from the label file
            with open(os.path.join(label_dir, label_file), 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.strip().split()
                class_id, x_center, y_center, width, height = map(float, parts)

                # Convert normalized positions to absolute (pixel) positions
                abs_x_center = x_center * WIDTH
                abs_y_center = y_center * HEIGHT
                abs_width = width * WIDTH
                abs_height = height * HEIGHT

                # Convert to COCO format (x_min, y_min, width, height)
                x_min = abs_x_center - (abs_width / 2)
                y_min = abs_y_center - (abs_height / 2)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id), 
                    "bbox": [x_min, y_min, abs_width, abs_height],
                    "area": abs_width * abs_height,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    # Write out the COCO dataset
    with open(output_path, 'w') as json_file:
        json.dump(coco_data, json_file)


if __name__ == '__main__':

    Convert2coco(
        label_dir="./datasets/dataset/train_labels/",
        output_path="./datasets/dataset/annotations/instances_train.json"
    )

    Convert2coco(
        label_dir="./datasets/dataset/val_labels/",
        output_path="./datasets/dataset/annotations/instances_val.json"
    )
