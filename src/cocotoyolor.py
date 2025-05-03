import os
import json
from pathlib import Path

# Set paths
coco_json_path = 'data/datection/labels/_annotations.coco.json'
output_labels_dir = Path('data/datection/labels/')
output_images_dir = Path('data/datection/images')

# Make sure output directories exist
output_labels_dir.mkdir(parents=True, exist_ok=True)
output_images_dir.mkdir(parents=True, exist_ok=True)

# Load COCO annotations
with open(coco_json_path) as f:
    data = json.load(f)

# Build image ID to filename map
image_map = {img['id']: img for img in data['images']}

# Create label files
for ann in data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # [x_min, y_min, width, height]

    # Get image details
    image_info = image_map[image_id]
    img_w, img_h = image_info['width'], image_info['height']
    img_name = image_info['file_name']
    stem = Path(img_name).stem

    # Convert to YOLO format
    x_center = (bbox[0] + bbox[2] / 2) / img_w
    y_center = (bbox[1] + bbox[3] / 2) / img_h
    width = bbox[2] / img_w
    height = bbox[3] / img_h

    yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    # Save to corresponding .txt file
    label_path = output_labels_dir / f"{stem}.txt"
    with open(label_path, 'a') as label_file:
        label_file.write(yolo_line)

    # Optionally move image files
    original_img_path = Path('detection/labels') / img_name
    target_img_path = output_images_dir / img_name
    if original_img_path.exists():
        target_img_path.write_bytes(original_img_path.read_bytes())
