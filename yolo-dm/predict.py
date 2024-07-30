from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/DCNv4-FASSF-100/weights/best.pt")  # load a custom model
# model = YOLO("runs/detect/yolov8n-100/weights/best.pt")  # load a custom model

# Predict with the model
results = model("test.JPG", save=True)  # predict on an image