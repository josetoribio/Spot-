#use run 4
from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('yolov8n.pt')  # load a custom model

# Predict with the model
results = model('Dog.jpeg')  # predict on an image

i=0
while i < len(results):
   # print(results[i])
    i += 1

print(results[0].speed)
split = results[0].speed

split = split(",",3)

print(split)

#print(type(results))
