from ultralytics import YOLO
import common as cmm
from PIL import Image

file = "khaisinh.jpeg"
weightPath = cmm.get_current_directory() + "\\runs\\segment\\train20\\weights"
# Load a model
model = YOLO('yolov8m-seg.pt')  # load an official model
model = YOLO(weightPath + '/best.pt')  # load a custom model

source = Image.open(cmm.get_current_directory() + "/" + file)

# Predict with the model
results = model(source, show=True, show_boxes=True)  # predict on an image
# View results
for r in results:
    print(r.boxes)
# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save(file + '_results.jpg')  # save image