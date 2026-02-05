# Siamese Face Recognition using FaceNet + MTCNN

A real-time **Siamese Neural Network–based face verification system** using **FaceNet embeddings** and **MTCNN face detection**. The system compares a reference face with a live webcam feed using **cosine similarity** to determine whether both faces belong to the same person.

---

## 🚀 Features

* Siamese-style face verification (embedding comparison)
* Face detection using **MTCNN**
* Face embedding using **FaceNet**
* Real-time webcam verification
* Cosine similarity–based decision
* No classical training required

---

## 🧠 How It Works

1. A reference image is processed to extract a face embedding
2. Webcam frames are captured in real time
3. Faces are detected and embedded
4. Embeddings are compared using cosine similarity
5. If similarity exceeds a threshold, faces are considered the same person

This follows the **Siamese Network paradigm**, where similarity between two inputs is learned via a shared network.

---

## 📁 Project Structure

```
project/
├── Mukhtar/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── face_recognition.py
└── README.md
```

---

## ⚙️ Requirements

* Python 3.9 – 3.12 (recommended)
* OpenCV
* NumPy
* MTCNN
* keras-facenet
* TensorFlow (backend)

---

## 📦 Installation

```bash
pip install opencv-python numpy mtcnn keras-facenet tensorflow
```

> ⚠️ Python 3.13 may show TensorFlow warnings. Python 3.10 or 3.11 is recommended.

---

## ▶️ Usage

1. Place reference images inside the person’s folder (e.g., `project/Mukhtar/`)
2. Update the reference image path in the script
3. Run the program:

```bash
python face_recognition.py
```

4. Press **Q** to exit the webcam window

---

## 🎯 Similarity Thresholds

| Cosine Similarity | Result                        |
| ----------------- | ----------------------------- |
| > 0.60            | Same person (high confidence) |
| 0.50 – 0.60       | Likely same                   |
| < 0.50            | Different person              |

Thresholds may be adjusted depending on lighting and camera quality.

---

## ✅ Advantages of Siamese FaceNet

* No retraining for new users
* High accuracy
* Scales easily to many identities
* Robust to lighting and pose variations

---

## ⚠️ Limitations

* Sensitive to very low-light conditions
* No anti-spoofing (photo/video attacks)
* Single-face comparison per frame

---

## 🔮 Future Improvements

* Multi-face recognition
* Embedding database with multiple users
* Anti-spoofing detection
* FastAPI / Streamlit deployment
* GPU acceleration

---

## 📚 References

* FaceNet: A Unified Embedding for Face Recognition
* MTCNN: Joint Face Detection and Alignment
* Siamese Neural Networks for One-Shot Learning

---

## 👨‍💻 Author

**Syed Mukhtar**

---

## 📜 License

This project is for educational and research purposes.
