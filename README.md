# Siamese Face Verification using FaceNet + MTCNN

This project implements a **Siamese Neural Network–based face verification system** using **MTCNN** for face detection and **FaceNet** for deep face embeddings. Two face images are compared using **cosine similarity** to determine whether they belong to the same person.

---

## 🚀 Features

* Siamese-style face comparison
* Accurate face detection with MTCNN
* Face embeddings using FaceNet
* Cosine similarity–based verification
* No model training required

---

## 🧠 How It Works

1. Two input images are loaded
2. Faces are detected and cropped using MTCNN
3. Each face is converted into a 128-D embedding using FaceNet
4. Embeddings are compared using cosine similarity
5. A threshold determines same or different person

---

## 📁 Project Structure

```
project/
├── John Cena/
│   └── john1.jpg
├── Seth Rollins/
│   └── seth2.jpg
└── face_recognition.py
```

---

## ⚙️ Requirements

* Python 3.9 – 3.12 (recommended)
* OpenCV
* NumPy
* MTCNN
* keras-facenet
* TensorFlow

---

## 📦 Installation

```bash
pip install opencv-python numpy mtcnn keras-facenet tensorflow
```

---

## ▶️ Usage

1. Place images inside their respective folders
2. Update image paths in the script if needed
3. Run the program:

```bash
python face_recognition.py
```

---

## 🎯 Similarity Threshold

| Cosine Similarity | Result           |
| ----------------- | ---------------- |
| > 0.50            | Same person      |
| ≤ 0.50            | Different person |

Threshold can be tuned based on image quality.

---

## ✅ Advantages

* Simple and lightweight
* High accuracy with deep embeddings
* Suitable for one-shot face verification

---

## ⚠️ Limitations

* Sensitive to lighting and pose variations
* No anti-spoofing protection
* Assumes one face per image

---

## 📚 References

* FaceNet: A Unified Embedding for Face Recognition
* MTCNN: Joint Face Detection and Alignment
* Siamese Neural Networks

---

## 👨‍💻 Author

**Syed Mukhtar**

---

## 📜 License

This project is intended for educational and research purposes.
