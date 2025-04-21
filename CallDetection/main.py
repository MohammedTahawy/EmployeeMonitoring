import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("yolo11n.pt")  # load an official model
mobile_phone_class = "cell phone"

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()

    for result in results[0].boxes:
        # Check if the detected object is a mobile phone
        if result.cls == mobile_phone_class or model.names[int(result.cls)] == mobile_phone_class:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            # Draw a rectangle around the mobile phone
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break