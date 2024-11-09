from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')
# Предсказание на изображении или папке с изображениями
results = model.predict(source='result', save=True, conf=0.5, save_txt=True)


