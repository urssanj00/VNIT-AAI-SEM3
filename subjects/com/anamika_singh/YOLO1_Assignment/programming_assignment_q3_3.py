from ultralytics import YOLO
import glob

# 1. Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# 2. Train the model on your custom dataset for 50 epochs
model.train(data='custom_data.yaml', epochs=50)

# 3. List of image paths for inference
image_paths = glob.glob('custom_dataset/*')

# 4. Run inference and save results to results/3_inference/
results = model.predict(
    image_paths,
    save=True,                # Save images with bounding boxes
    project='results',        # Top-level results folder
    name='3_inference',       # Subfolder for this run
    exist_ok=True             # Overwrite if the folder exists
)

# 5. Print detected boxes, labels, and confidence scores
for result in results:
    for box in result.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]      # bounding box coordinates
        conf = box.conf.cpu().numpy()[0]      # confidence score
        cls = int(box.cls.cpu().numpy()[0])   # class index
        label = model.names[cls]              # class label
        print(f"Label: {label}, Confidence: {conf:.2f}, Box: {xyxy}")

# 6. Run validation on your dataset YAML file
metrics = model.val(data='custom_data.yaml')

# 7. Print evaluation metrics safely
try:
    print(f"2. Recall: {metrics.box.mr():.4f}")
    print(f"2. mAP@0.5 (map50): {metrics.box.map50():.4f}")
    print(f"4. mAP@0.5:0.95 (map): {metrics.box.map():.4f}")
    print(f"1. Precision: {metrics.box.mp():.4f}")
except Exception as e:
    print("Metrics could not be calculated. Check if your validation set has labels.")
    print(f"Error: {e}")
