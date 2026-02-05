import cv2
import os
import numpy as np

people = ["Elon Musk","John Cena","Robert Downey","Seth Rollins","Steve Rogers"]
# Use absolute path or ensure DIR is correct
DIR = r'project'

# Use built-in path to avoid "XML not found" errors
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue
            
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            
            # Check if image was read correctly
            if img_array is None:
                continue
                
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

# Convert to numpy arrays for compatibility with recognizers
features = np.array(features, dtype='object')
labels = np.array(labels)

print(f"Length of features = {len(features)}")
print(f"Length of labels obtained = {len(labels)}")

# Example: How to initialize a recognizer in 2026 (requires opencv-contrib-python)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save("face_trained.yml")
np.save("features.npy",features,allow_pickle=True)
np.save("labels.npy",labels,allow_pickle=True)

