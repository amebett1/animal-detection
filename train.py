import os
from ultralytics import YOLO

if __name__ == '__main__':
    
    config_path = './config.yaml'

    # Load model
    model = YOLO('yolov8n.pt')

    # Train model
    model.train(data=config_path, epochs=200, batch=32, name='animal_detector_v8')