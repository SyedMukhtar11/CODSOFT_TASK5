import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

detector = MTCNN()
embedder = FaceNet()

def detect_and_crop(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]['box']
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    return face

def get_embedding(face):
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

# Load faces
face1 = detect_and_crop("project/John Cena/john1.jpg")
face2 = detect_and_crop("project/Seth Rollins/seth2.jpg")

emb1 = get_embedding(face1)
emb2 = get_embedding(face2)

# Cosine similarity
similarity = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

print("Similarity Score:", similarity)

if similarity > 0.5:
    print("✅ Same Person")
else:
    print("❌ Different Person")
