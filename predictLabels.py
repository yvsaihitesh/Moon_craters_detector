<<<<<<< HEAD
import os
from ultralytics import YOLO
from PIL import Image

# File Paths
test_dir = r"testImages"
label_dir = r"predictedLabels"
os.makedirs(label_dir, exist_ok=True)

# Loading the best model
model = YOLO(r"best.pt")

conf_threshold = 0.25
iou_threshold = 0.5

# Runing prediction on all test images
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_dir, filename)
        img = Image.open(image_path)
        width, height = img.size
        
        results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
        result = results[0]
        boxes = result.boxes
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        
        with open(label_path, "w") as f:
            for box in boxes:
                cls = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print(" Label generation complete.")
=======
import os
from ultralytics import YOLO
from PIL import Image

# File Paths
test_dir = r"testImages"
label_dir = r"predictedLabels"
os.makedirs(label_dir, exist_ok=True)

# Loading the best model
model = YOLO(r"best.pt")

conf_threshold = 0.25
iou_threshold = 0.5

# Runing prediction on all test images
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_dir, filename)
        img = Image.open(image_path)
        width, height = img.size
        
        results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
        result = results[0]
        boxes = result.boxes
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        
        with open(label_path, "w") as f:
            for box in boxes:
                cls = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print(" Label generation complete.")
>>>>>>> 2576b1976c9b04e1e2ec5e64a90b132f8094ea12
