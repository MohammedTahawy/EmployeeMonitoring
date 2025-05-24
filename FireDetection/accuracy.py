import os
import cv2
import math
from ultralytics import YOLO

# Load the YOLO fire detection model
model = YOLO('fire.pt')

# Path to the dataset folder containing only fire images
dataset_path = 'datasets/val'

# Initialize counters
correct_predictions = 0
total_samples = 0

# Iterate through dataset images
for image_name in os.listdir(dataset_path):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        total_samples += 1
        image_path = os.path.join(dataset_path, image_name)

        # Read the image
        image = cv2.imread(image_path)

        # Run the model on the image
        results = model(image)

        # Check if fire is detected
        fire_detected = False
        for result in results[0].boxes:
            confidence = math.ceil(result.conf[0] * 100)
            if confidence > 20:  # Confidence threshold
                fire_detected = True
                break

        # Since all images are fire, fire_detected should be True
        if fire_detected:
            correct_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / total_samples) * 100
print(f"Accuracy: {accuracy:.2f}%")