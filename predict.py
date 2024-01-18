from ultralytics import YOLO
import common as cmm

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(cmm.get_pva_yolo_home() + '/best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image