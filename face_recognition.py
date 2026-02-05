import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

detector = MTCNN()
embedder = FaceNet()

# ---------------------------
# Detect & crop from IMAGE
# ---------------------------
def detect_and_crop_from_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']
    x, y = abs(x), abs(y)   # fix negative coords

    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    return face

# ---------------------------
# Get FaceNet embedding
# ---------------------------
def get_embedding(face):
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

# ---------------------------
# Load reference face
# ---------------------------
ref_img = cv2.imread("project/Mukhtar/WhatsApp Image 2026-02-05 at 9.43.28 PM.jpeg")  # <-- image file
ref_face = detect_and_crop_from_image(ref_img)

if ref_face is None:
    raise ValueError("No face found in reference image")

ref_emb = get_embedding(ref_face)

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_and_crop_from_image(frame)

    if face is not None:
        emb = get_embedding(face)

        similarity = np.dot(ref_emb, emb) / (norm(ref_emb) * norm(emb))

        label = "Same Person" if similarity > 0.5 else "Different Person"
        color = (0, 255, 0) if similarity > 0.5 else (0, 0, 255)

        cv2.putText(
            frame,
            f"{label} ({similarity:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
