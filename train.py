from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8m-seg.yaml').load('yolov8m-seg.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='C:\\Study\\MachineLearning\\birth\\datasets\\birth.v2i.yolov5pytorch\\data.yaml', epochs=200, imgsz=640, workers=0)