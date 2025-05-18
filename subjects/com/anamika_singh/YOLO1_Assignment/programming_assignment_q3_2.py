from ultralytics import YOLO
import glob
import pandas as pd
# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# List of image paths
image_paths = glob.glob('q3_2_*.jpeg')

# Run inference and save results to results/yolo_inference/
results = model.predict(
    image_paths,
    save=True,                 # Save images with bounding boxes
    project='results',          # Top-level results folder
    name='yolo_inference',     # Subfolder for this run
    exist_ok=True              # Overwrite if the folder exists
)

for result in results:
    # Show the results in a window (requires GUI support)
    result.show()


# To print detected boxes, labels, and confidence scores:
for result in results:
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy()[0]  # bounding box coordinates (x_min, y_min, x_max, y_max)
        conf = box.conf.cpu().numpy()[0]  # confidence score
        cls = int(box.cls.cpu().numpy()[0])  # class index
        label = model.names[cls]  # class label
        print(f"Label: {label}, Confidence: {conf:.2f}, Box: {xyxy}")
