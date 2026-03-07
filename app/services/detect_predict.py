import os
import cv2
import numpy as np
import tensorflow as tf

# 1. Dynamically find absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASCADE_PATH = os.path.join(BASE_DIR, 'models', 'haarcascade-frontalface-default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'my_model_emotion_detection_1.keras')

# 2. Load models
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_model = tf.keras.models.load_model(MODEL_PATH)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

def analyze_emotion(image_path):
    """Function used by the FastAPI (main.py) to return JSON results."""
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "message": "Impossible de lire l'image."}
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return {"status": "success", "faces": []}
        
    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
        
        prediction = emotion_model.predict(roi_reshaped, verbose=0)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        confidence = float(np.max(prediction)) * 100
        
        results.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "emotion": emotion,
            "confidence": confidence
        })
        
    return {"status": "success", "faces": results}