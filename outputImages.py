from ultralytics import YOLO
import os

# Loading the trained best YOLO model
model = YOLO(r"best.pt")

# File Paths
input_folder = r"testImages"
output_folder = r"predictedImages"
os.makedirs(output_folder, exist_ok=True)

# Running inference on each image
for img_file in os.listdir(input_folder):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, img_file)
        results = model(img_path)

        results[0].save(filename=os.path.join(output_folder, img_file))
