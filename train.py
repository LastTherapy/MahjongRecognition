from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")
# Запуск обучения
model.train(data='data.yaml', epochs=50, imgsz=640)
