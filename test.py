from ultralytics import YOLO

model = YOLO('runs/detect/animal_detector_v10/weights/best.pt')


model.predict(source='1115.mp4', save=False, conf=0.5, show=True)