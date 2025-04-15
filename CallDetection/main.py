import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(2)
model = YOLO("yolo11n.pt")  # load an official model

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break