import os
from ultralytics import YOLOv10


config_path = './config.yaml'

#load model 

model = YOLOv10("yolov10n.pt")

#train model
model.train(data=config_path, epochs=200, batch=32)
