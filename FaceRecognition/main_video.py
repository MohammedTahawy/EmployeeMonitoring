from typing import Union, Any
import cv2
from simple_facerec import SimpleFacerec
import time
import sqlite3

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


cap = cv2.VideoCapture(0)

# Connect to SQLite database
conn = sqlite3.connect('EmployeeMonitoring.db')
cursor = conn.cursor()

# Create events table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER,
    person_id TEXT,
    time TEXT
)
''')
conn.commit()

# Dictionary to store the last detection time of each person
last_detection: dict[Union[str, Any], float] = {}

while True:
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
            # Insert event into the database
            cursor.execute('''
            INSERT INTO events (camera_id, person_id, time)
            VALUES (?, ?, ?)
            ''', (2, name, current_time))
            conn.commit()

            # Update the last detection time
            last_detection[name] = time.time()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()