# EmoSense 🎭
### Real-Time Facial Emotion Detection System

> CNN-based emotion recognition from live webcam feed — trained on FER-2013, deployed with FastAPI and OpenCV at ~25 FPS.

---

## 📌 Overview

EmoSense is a deep learning system that detects **7 facial emotions in real time** from a live webcam stream. It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and deploys inference through an OpenCV pipeline backed by a FastAPI server.

| Property | Detail |
|---|---|
| **Model** | Custom CNN (TensorFlow / Keras) |
| **Dataset** | FER-2013 (35,887 images, 7 emotion classes) |
| **Validation Accuracy** | 64.7% |
| **Inference Speed** | ~25 FPS (real-time) |
| **Backend** | FastAPI + WebSocket |
| **Frontend** | OpenCV webcam pipeline |

---

## 🎯 Emotion Classes

The model detects the following 7 emotions:

| Label | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## 🗂️ Project Structure

```
EmoSense/
├── model/
│   ├── train.py              # CNN training script
│   ├── model.py              # Model architecture definition
│   └── emosense_model.h5     # Saved trained model weights
├── backend/
│   ├── main.py               # FastAPI server with WebSocket support
│   └── inference.py          # Real-time prediction pipeline
├── frontend/
│   └── index.html            # Browser-based webcam UI
├── data/
│   └── fer2013.csv           # FER-2013 dataset (download separately)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/bipanshu1007/emosense.git
cd emosense
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the FER-2013 dataset
Download `fer2013.csv` from [Kaggle — FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and place it inside the `data/` folder.

---

## 🏋️ Training the Model

```bash
python model/train.py
```

This will:
- Load and preprocess the FER-2013 dataset
- Train the CNN for the configured number of epochs
- Save the best model weights to `model/emosense_model.h5`
- Print validation accuracy and loss curves

**Training configuration (defaults):**

| Parameter | Value |
|---|---|
| Input size | 48 × 48 grayscale |
| Batch size | 64 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 50 |
| Validation accuracy | ~64.7% |

---

## 🚀 Running the App

### Start the FastAPI backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Open the frontend
Open `frontend/index.html` in your browser, or visit:
```
http://localhost:8000
```

The webcam feed will start automatically and display the detected emotion label overlaid on the face in real time at **~25 FPS**.

---

## 🧠 Model Architecture

```
Input (48×48×1)
    │
    ▼
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
    ▼
Flatten → Dense(256) → ReLU → Dropout(0.5)
    │
    ▼
Dense(7) → Softmax
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~72% |
| Validation Accuracy | **64.7%** |
| Inference Speed | **~25 FPS** |
| Dataset Size | 35,887 images |
| Classes | 7 emotions |

> Note: FER-2013 is a challenging benchmark — human-level accuracy on FER-2013 is reported at ~65%, making this model competitive with human performance.

---

## 📦 Requirements

```
tensorflow>=2.10
keras
opencv-python
fastapi
uvicorn
websockets
numpy
pandas
matplotlib
scikit-learn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] Fine-tune with EfficientNetB0 for higher accuracy
- [ ] Add multi-face detection support
- [ ] Deploy to cloud (AWS / Heroku)
- [ ] Add emotion history graph in the UI
- [ ] Export model to TensorFlow Lite for mobile deployment

---

## 👤 Author

**Bipanshu Kashyap**
B.Tech Computer Science — Graphic Era University, Dehradun
- GitHub: [@bipanshu1007](https://github.com/bipanshu1007)
- LinkedIn: [bipanshu-kashyap-925785321](https://www.linkedin.com/in/bipanshu-kashyap-925785321/)
- Email: kashyapbipanshu@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

<p align="center">Made with ❤️ by Bipanshu Kashyap</p>
