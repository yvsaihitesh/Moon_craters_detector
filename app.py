from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Loading the trained best model
model = YOLO("best.pt")

# Class colors
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for crater
    1: (255, 0, 255)   # Magenta for boulder
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    input_path = 'input.jpg'
    file.save(input_path)

    results = model(input_path)[0]

    img = cv2.imread(input_path)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    output_path = os.path.join('static', 'output.jpg')
    cv2.imwrite(output_path, img)

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))