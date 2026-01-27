import cv2
import numpy as np
import tensorflow as tf

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    face = face / 255.0
    return face
def embedding_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    return model
input_a = tf.keras.Input(shape=(100,100,3))
input_b = tf.keras.Input(shape=(100,100,3))

base_network = embedding_model()

emb_a = base_network(input_a)
emb_b = base_network(input_b)

# L1 Distance
distance = tf.keras.layers.Lambda(
    lambda tensors: tf.abs(tensors[0] - tensors[1])
)([emb_a, emb_b])

output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)

siamese_model = tf.keras.Model([input_a, input_b], output)

siamese_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

siamese_model.summary()
img1 = detect_and_crop("project/John Cena/john1.jpg")
img2 = detect_and_crop("project/John Cena/john4.jpg")

img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)

similarity = siamese_model.predict([img1, img2])[0][0]

print("Similarity Score:", similarity)

if similarity > 0.5:
    print("✅ Same Person")
else:
    print("❌ Different Person")
