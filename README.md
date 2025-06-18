# 🌑 Moon Crater & Boulder Detection using Custom CNN

This project is to detect craters and boulders on high-resolution lunar surface images using a Convolutional Neural Network (CNN) based yolov10s model — optimized for high accuracy. This includes an interactive web UI powered by Flask for real-time predictions.

---

## 🚀 Project Overview

- **Goal:** Detect and classify craters (class `0`) and boulders (class `1`) in lunar surface images.
- **Model:** Yolov10s(CNN Based).
- **Frontend:** HTML/CSS/Javascript - based web interface.
- **Backend:** Python (Flask) to process image uploads and display detections output.
- **Evaluation Metrics:** `mAP@0.5`, `mAP@0.5:0.95`, `Precision`, `Recall`, `F1 Score`, and `Accuracy`.

---

## 🧠 Model Architecture

- YOLO-style grid prediction.
- Model designed for detecting small objects with high spatial accuracy.
- Trained on lunar datasets with bounding boxes for craters.

---

## 🗂️ Folder Structure

```
moon-craters-detector/
│
├── static/  
|    └── moon.glb             # 3D moon model for website
├── templates/                # HTML template for Flask frontend
│   └── index.html
│
├── lunar_subset/             # Dataset directory ( Images and Labels )
|    └── images
|        ├── train
|        └── valid
|    └── labels
|        ├── train
|        └── valid
│
├── testImages                # Test images
├── train.py                  # Model training code
├── outputImages.py           # Code to get detected craters images of test sample
├── predictLabels             # Code to get predicted labels
├── app.py                    # Flask web server
├── requirements.txt          # All Python dependencies
└── README.md                 # This file
```

---

## 🧪 Sample Results

- ✅ Detected bounding boxes with class (0 = crater, 1 = boulder)
- 📊 Evaluation metrics printed after each epoch
- 🖼️ Flask UI for user-uploaded image inference

---

## 🛠️ How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/moon-craters-detector.git
cd moon-craters-detector
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train.py
```

### 4. Visualize Detected Craters Images
```bash
python outputImages.py
```

### 5. Visualize Detected Craters Labels
```bash
python predictLabels.py
```

### 5. Launch the Web App
```bash
python app.py
```
Then open your browser at: [http://localhost:5000](http://localhost:5000)

---

## 📈 Evaluation Metrics (Displayed Per Epoch)

- `mAP@0.5`
- `mAP@0.5:0.95`
- `Precision`
- `Recall`
- `F1 Score`
- `Accuracy`

---

## 📷 Dataset Info

- Custom lunar images with labeled crater bounding boxes.
- Supports `.txt` YOLO format bounding box format.
- Resolution: High-res lunar surface images.

---

## 📌 Notes

- Model outputs 0 for craters and 1 for boulders. Green Boxes for craters and Magenta Boxes for boulders.
- Grid-based prediction, optimized for small object detection.
- Built using Yolov10s.

---

## 📚 Future Work

- Improve small-object detection under extreme lighting conditions.
- Expand to 3D terrain mapping and detection using elevation data.
- Improve speed with quantization/pruning techniques while preserving accuracy.

---

## 🧑‍💻 Members

- **Yacha Venkata Sai Hitesh**
- **Bandikatla Vivek Kumar**
- **Dontamsetti Kedar Subrahmanya Rishi**