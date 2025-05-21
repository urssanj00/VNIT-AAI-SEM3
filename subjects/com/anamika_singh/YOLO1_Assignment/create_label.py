from ultralytics import YOLO
import glob
import os

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Define your custom dataset folder
image_dir = '/Sanjeev/VNIT_CLASSES/VNIT-AAI-SEM3/custom_dataset_yolo/train_1'
image_paths = glob.glob(os.path.join(image_dir, '*'))

# Run inference
results = model.predict(
    image_paths,
    save=False,  # Don't save images with bounding boxes
    verbose=False  # Suppress verbose output
)

# Process each result
for i, result in enumerate(results):
    image_path = image_paths[i]  # Get the corresponding image path
    label_file_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    label_file_path = os.path.join(image_dir, label_file_name)

    with open(label_file_path, 'w') as f:
        has_objects = False
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            cls = int(box.cls.cpu().numpy()[0])
            label = model.names[cls]

            # Convert bounding box to YOLO format
            x_center = (xyxy[0] + xyxy[2]) / (2 * result.orig_shape[1])
            y_center = (xyxy[1] + xyxy[3]) / (2 * result.orig_shape[0])
            width = (xyxy[2] - xyxy[0]) / result.orig_shape[1]
            height = (xyxy[3] - xyxy[1]) / result.orig_shape[0]

            # Write to label file
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            has_objects = True

        if not has_objects:
            print(f"No objects detected in {image_path}, created empty label file.")
        else:
            print(f"Label file created for {image_path}")
