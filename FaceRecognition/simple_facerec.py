import face_recognition
import cv2
import os
import pickle
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.2

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """

        # Check if encodings file exists
        encodings_file = "FaceRecognition/images/encodings.pkl"
        if os.path.exists(encodings_file):
            # Load encodings from file
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
            print("Loaded encodings from file.")

        else:
            # Encode images and save to file
            print("Encoding images...")
            # Loop through each person in the training directory
            for root, dirs, files in os.walk(images_path):
                for file in files:
                    if file.endswith(('jpg', 'jpeg', 'png')):
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path)
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Get the name of the person
                        name = os.path.basename(root)

                        # Encode the loaded image into a feature vector
                        img_encoding = face_recognition.face_encodings(rgb_img)[0]

                        # Store the encoding and the name
                        self.known_face_encodings.append(img_encoding)
                        self.known_face_names.append(name)
            print("Encoding images loaded")

            with open(encodings_file, "wb") as f:
                data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
                pickle.dump(data, f)


    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding ,tolerance=0.35)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
