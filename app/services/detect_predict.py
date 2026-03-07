import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from app.config import CASCADE_PATH, MODEL_SAVE_PATH, class_names

# --- Load Models Globally ---
# Load once at startup, not per request
try:
    if not os.path.exists(CASCADE_PATH):
        print(f"⚠️ Cascade file not found at {CASCADE_PATH}")
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"⚠️ Keras model not found at {MODEL_SAVE_PATH}")
        
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    emotion_model = load_model(MODEL_SAVE_PATH)
    print("✅ AI Models loaded successfully into memory.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    face_cascade = None
    emotion_model = None

def analyze_emotion(image):
    """
    Analyzes an OpenCV image and returns emotion data.
    """
    if face_cascade is None or emotion_model is None:
        return {"status": "error", "message": "Models not loaded"}

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 5)

    if len(faces) == 0:
        return {"status": "error", "message": "Aucun visage detecte", "faces": []}

    results = []
    # Currently processing only the first face to match existing logic
    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]
    resized_face = cv2.resize(face_roi, (48, 48))
    
    processed_face = np.expand_dims(resized_face, axis=-1)
    processed_face = np.expand_dims(processed_face, axis=0)
    processed_face = processed_face / 255.0  # Normalized as in training

    prediction = emotion_model.predict(processed_face, verbose=0)
    score = float(np.max(prediction))
    emotion_index = int(np.argmax(prediction))
    emotion_label = class_names[emotion_index]

    return {
        "status": "success",
        "emotion": emotion_label,
        "score": round(score * 100, 2),
        "faces": [{"bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}}]
    }
