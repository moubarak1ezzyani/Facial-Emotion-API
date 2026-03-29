# 😐 Facial Emotion Detection API

## 📄 Project Overview
This project is a prototype REST API developed for a UX analysis startup. It combines **Computer Vision** and **Deep Learning** to analyze facial emotions on static images and store the results for statistical studies. 

The application exposes a high-performance, asynchronous FastAPI server capable of detecting a face in an uploaded image, classifying its emotion into one of 7 categories (*Happy, Sad, Angry, Surprise, Neutral, Fear, Disgusted*), and archiving the data.

---

## ⚙️ Technical Architecture & Pipeline


[Image of Convolutional Neural Network architecture]


The data processing pipeline follows these rigorous steps:
1. **Reception**: The API receives an image via the `/predict_emotion` endpoint.
2. **Detection (OpenCV)**: A *Haar Cascade* classifier isolates the face within the image.
3. **Normalization**: The face is cropped, converted to grayscale, resized to 48x48 pixels, and scaled to a [0-1] range.
4. **Inference (CNN)**: A TensorFlow (`.keras`) model predicts the emotion and assigns a confidence score.
5. **Persistence (SQLAlchemy Async)**: A non-blocking save is executed into a PostgreSQL database.

---

## 📂 Repository Structure

```bash
Facial-Emotion-API/
├── .github/
│   └── workflows/
│       └── python-ci.yml                # CI/CD automated testing pipeline
├── app/                                 
│   └── services/                        # Core API and ML Logic
│       ├── detect_predict.py            # AI inference service
│       ├── config.py                    # Environment variables mapping
│       ├── database.py                  # Async SQLAlchemy engine
│       ├── main.py                      # FastAPI application and routes
│       └── models.py                    # PostgreSQL table schemas
├── data/
│   ├── Data_Kaggle_Emotional_Detection/ # Original dataset
│   └── samples/                         # Test images for each emotion
├── models/
│   ├── haarcascade-frontalface-default.xml # Face detection weights
│   ├── my_model_emotion_detection_1.keras  # CNN Model Version 1
│   └── my_model_emotion_detection_2.keras  # CNN Model Version 2
├── notebooks/                           # Jupyter notebooks for EDA
├── output/                              # Generated visualizations
├── scripts/
│   └── training.py                      # CNN Training script
├── tests/
│   └── test_unitaire.py                 # Pytest suite
├── .env                                 # Environment credentials (ignored by Git)
├── create_tables.py                     # DB initialization script
├── detect_and_predict.py                # Standalone script for visual testing
└── README.md                            # Documentation
```
---

## 🚀 Installation & Setup

### 1. Prerequisites

* Python 3.9+
* PostgreSQL installed and running locally.

### 2. Clone and Install

Clone the repository and install the required Python libraries:

```bash
git clone [https://github.com/votre-user/Facial-Emotion-API.git](https://github.com/votre-user/Facial-Emotion-API.git)
pip install -r requirements.txt

```

### 3. Database Configuration

Create a `.env` file at the root of the project to securely store your credentials:

```env
DB_USER=postgres
DB_PASS=votre_mot_de_passe
DB_HOST=localhost
DB_NAME=emotion_db

```

Initialize the tables in your PostgreSQL database by running the setup script from the root folder:

```bash
python create_tables.py

```

*(This automatically creates the `EmotionTable` via SQLAlchemy).*

---

## 💻 Usage

### Run the API Server

Start the Uvicorn server with auto-reload enabled (adjust path based on your exact `main.py` location):

```bash
uvicorn app.services.main:app --reload 

```

The API will be accessible at: `http://127.0.0.1:8000`

### Run the Standalone Script

To test the detection and prediction visually on a local image from the `data/samples/` folder without starting the server:

```bash
python detect_and_predict.py

```

*(Ensure you modify the `image_path` variable inside the script before running).*

---

## 📡 API Endpoints

Interactive Swagger UI documentation is automatically available at `http://127.0.0.1:8000/docs`.

### 1️⃣ `POST /predict_emotion`

Analyzes an uploaded image.

* **Input**: Image file (`UploadFile`).
* **Process**: Detection -> Prediction -> DB Save.
* **Output Example (JSON)**:
```json
{
  "emotion": "happy",
  "confidence": 98.5,
  "saved_id": 1,
  "all_faces_detected": [...]
}

```



### 2️⃣ `GET /history`

Retrieves the history of all analyzed images stored in the database.

* **Output**: List of entries including ID, File Name, Emotion, Confidence score, and Date created.

---

## ✅ Testing & Quality Assurance

This project includes automated unit tests via **Pytest** and continuous integration via **GitHub Actions** (`python-ci.yml`) to ensure the robustness of the ML pipeline.

Run the test suite manually:

```bash
pytest tests/test_unitaire.py

```

**Test Coverage Includes:**

* `test_model_save_and_load`: Verifies the integrity of saving and loading the `.keras` model.
* `test_prediction_format`: Ensures the CNN output correctly returns a tensor with the shape `(1, 7)`.


