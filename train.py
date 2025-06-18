import torch
from torch.serialization import add_safe_globals
from torch.nn import Sequential
from ultralytics.nn.tasks import DetectionModel

# Allow these classes for safe model loading
add_safe_globals([Sequential, DetectionModel])
from ultralytics import YOLO

# Function to print metrics after every epoch to evaluate
def print_metrics_callback(trainer):
    metrics = trainer.metrics
    mAP50 = metrics.get("metrics/mAP50(B)", 0)
    mAP5095 = metrics.get("metrics/mAP50-95(B)", 0)
    precision = metrics.get("metrics/precision(B)", 0)
    recall = metrics.get("metrics/recall(B)", 0)

    if hasattr(mAP50, "item"): mAP50 = mAP50.item()
    if hasattr(mAP5095, "item"): mAP5095 = mAP5095.item()
    if hasattr(precision, "item"): precision = precision.item()
    if hasattr(recall, "item"): recall = recall.item()
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Approximate Accuracy
    accuracy = (precision + recall) / 2

    print(f"\nEpoch Summary:")
    print(f"  ➤ mAP@0.5      : {mAP50:.4f}")
    print(f"  ➤ mAP@0.5:0.95 : {mAP5095:.4f}")
    print(f"  ➤ Precision    : {precision:.4f}")
    print(f"  ➤ Recall       : {recall:.4f}")
    print(f"  ➤ F1 Score     : {f1_score:.4f}")
    print(f"  ➤ Accuracy     : {accuracy:.4f} (approx)")


if __name__ == '__main__':
    # Loading YOLO model
    model = YOLO("yolov10s.pt")
model.train(
    data="lunar_subset/data.yaml",
    epochs=50,
    imgsz=640,
    device='cpu',
)