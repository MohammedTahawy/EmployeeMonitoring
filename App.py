import cv2
from FaceRecognition.simple_facerec import SimpleFacerec
from typing import Union, Any
import sqlite3
import time
# from ultralytics import YOLO
from yolov5 import YOLO
import cvzone
import math


sfr = SimpleFacerec()
sfr.load_encoding_images("FaceRecognition/images/")
cap = cv2.VideoCapture("SmokingDetection/Smoking_detection.mp4")
last_detection: dict[Union[str, Any], float] = {}

mobile_model = YOLO("CallDetection/yolo11n.pt")
mobile_phone_class = "cell phone"

fire_model = YOLO('FireDetection/fire.pt')
fire_classnames = ['fire']

smoking_model = YOLO('SmokingDetection/cigarette.pt')

while True:
    #FaceRecognition
    ret, frame = cap.read()
    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Check if the person was detected recently
        if name not in last_detection or (time.time() - last_detection[name]) > 60:  # 60-second threshold
            # Update the last detection time
            last_detection[name] = time.time()

    # CallDetection
    mobile_results = mobile_model(frame)
    # annotated_frame = frame.copy()
    for result in mobile_results[0].boxes:
        # Check if the detected object is a mobile phone
        if result.cls == mobile_phone_class or mobile_model.names[int(result.cls)] == mobile_phone_class:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            # Draw a rectangle around the mobile phone
            cvzone.putTextRect(frame, 'cell phone', [x1 , y1 - 10])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FireDetection
    result = fire_model(frame, stream=True)
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{fire_classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],scale=1.5,thickness=2)

    # SmokingDetection
    smoking_results = smoking_model(frame)
    for result in smoking_results[0].boxes:
        # Check if the detected object is a smoking person
        if result.cls == 0 or smoking_model.names[int(result.cls)] == 0:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            # Draw a rectangle around the mobile phone
            cvzone.putTextRect(frame, 'smoking', [x1 , y1 - 10])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
























    cv2.imshow("EmployeeMonitoring", frame )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break