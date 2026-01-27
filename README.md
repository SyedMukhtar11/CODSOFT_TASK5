Sure! Here's a **complete README.md** for your **Face Detection & Recognition Project** that covers both your Haar Cascade detection and the Siamese network approach.



# 🖼️ Face Detection & Recognition Project

This project demonstrates **face detection** using **Haar Cascades** and **face recognition** using **LBPH / Siamese networks** in Python with OpenCV and TensorFlow/Keras.

It includes:

1. Detecting faces in images.
2. Recognizing known people using **LBPH (Local Binary Patterns Histogram)**.
3. Recognizing known people using a **Siamese network** for similarity-based face recognition.



## 📂 Project Structure

```
├── haar_face.xml              # Pre-trained Haar cascade classifier
├── face_trained.yml           # Trained LBPH face recognizer
├── project/                   # Folder containing images of people
│   Elon Musk
    John Cena
    Robert Downey
    Seth Rollins
    Steve Rogers
├── face_detection.py          # Haar cascade detection code
├── face_recognition.py        # LBPH + Siamese face recognition code
├── README.md
```



## 🛠️ Requirements

* Python 3.8+
* OpenCV (`opencv-python` and `opencv-contrib-python`)
* TensorFlow / Keras (for Siamese network)
* NumPy

Install dependencies:

```bash
pip install opencv-python opencv-contrib-python tensorflow keras numpy
```



## 🔹 Face Detection using Haar Cascade

The `face_detection.py` script:

* Loads an image
* Converts it to grayscale
* Detects faces using Haar cascade (`haar_face.xml`)
* Draws rectangles around detected faces

```python
import cv2 as cv

img = cv.imread("project/seth2.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)
cv.waitKey(0)
```

✅ Detects faces in a group image and highlights them.



## 🔹 Face Recognition using LBPH

The `face_recognition.py` script:

* Loads a trained LBPH model (`face_trained.yml`)
* Detects faces using Haar cascade
* Predicts the label and confidence for each detected face
* Displays the name and bounding box

```python
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Detect faces, predict and display names
label, confidence = face_recognizer.predict(faces_roi)
```



## 🔹 Face Recognition using Siamese Network

* Uses a **Siamese network** to compare detected faces with reference images.
* Outputs similarity score for each person.
* Detects unknown faces if similarity is below a threshold.

```python
# Compare detected face with reference images
score = siamese_model.predict([face_processed, ref_img_processed])[0][0]
```



## 📝 Usage

1. **Detect faces**:

```bash
python face_detection.py
```

2. **Recognize faces (LBPH)**:

```bash
python face_recognition.py
```

3. **Recognize faces (Siamese network)**:

* Ensure `siamese_model.h5` is present.
* Update reference images for each person.
* Run the script.



## 🔧 Notes

* Haar Cascade works well for frontal faces but may fail on occluded or angled faces.
* LBPH works best for known, pre-trained faces.
* Siamese network is suitable for **few-shot recognition** and can generalize to unseen faces.
* Update **image paths** according to your folder structure.



## 🔮 Future Improvements

* Integrate **real-time face recognition** using webcam.
* Use **deep learning-based face detectors** like MTCNN or RetinaFace.
* Add **attention mechanism** for better Siamese performance.
* Deploy as a **web app** using Streamlit or Flask.



## 📜 References

* [OpenCV Haar Cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
* [LBPH Face Recognizer](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_face_detection.html)
* [Siamese Network for Face Recognition](https://www.tensorflow.org/tutorials/keras/siamese)



