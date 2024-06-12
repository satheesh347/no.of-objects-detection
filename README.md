import torch
import cv2
from matplotlib import pyplot as plt

# Load the pre-trained YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image from file
image_path = '/content/sample_data/g.jpeg'  # Replace with the path to your image
img = cv2.imread(image_path)

# Perform object detection
results = model(img)

# Extract the number of detected objects
detected_objects = results.pandas().xyxy[0]
num_objects = len(detected_objects)
print(f'Number of detected objects: {num_objects}')

# Optionally, display the image with detected objects
results.show()
