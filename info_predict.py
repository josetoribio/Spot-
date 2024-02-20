#https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

#use run 4
from ultralytics import YOLO
import cv2


# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model

#load picture
path = 'walking_dog.jpeg'
# Predict with the model
results = model(path)  # predict on an image

i=0
while i < len(results):
    print("2")
    print(results[i])

    i += 1

#safety procautions
    #ONLY if the object is a book 
    #get the bounding box 
    #and pick up the object
    
detected_object_name = int(results[0].boxes.cls[0]) # how to get name cls[number] to get name
x1, y1, x2, y2 = results[0].boxes.xyxy[0]
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
print(results[0].boxes.xyxy[0])
print("BoundingBoxes\nx1: "+str(x1)+"\ny1: "+str(y1)+"\nx2: "+str(x2)+ "\ny2: "+str(y2))
print((results[0].names[detected_object_name]))
print(int(results[0].boxes.cls[0])) #what class name 
print(len(results[0])) # the length of items detected 
