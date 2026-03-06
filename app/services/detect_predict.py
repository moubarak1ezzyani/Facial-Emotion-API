import cv2
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
import sys

# --- Fix import error by stepping up 3 levels ---
# 1. services -> 2. app -> 3. root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from app.config import CASCADE_PATH, MODEL_SAVE_PATH, EMOTIONS
# Suppress TensorFlow logging warnings for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 2. Load Models Globally ---
# We do this OUTSIDE the function so the API doesn't reload the heavy model
# on every single user request.
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    emotion_model = load_model(MODEL_SAVE_PATH)
    print("✅ AI Models loaded successfully into memory.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # If this fails, the API won't work, so it's good to know immediately.

def analyze_emotion(image_path):
    """
    Analyzes an image and returns a dictionary with bounding boxes and emotions.
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image not found: {image_path}"}

    # Read and convert image
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": "Could not read the image file."}
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return {"status": "success", "message": "No face detected.", "faces": []}

    results = []

    # Process each detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        
        # Normalize the image (matches the rescale=1./255 in training.py!)
        roi_normalized = roi_resized / 255.0
        
        # Reshape for the CNN (Batch, Height, Width, Channels)
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
        
        # Predict
        prediction = emotion_model.predict(roi_reshaped, verbose=0) # verbose=0 hides the progress bar
        max_index = int(np.argmax(prediction[0]))
        
        face_data = {
            "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "emotion": EMOTIONS[max_index],
            "confidence": round(float(prediction[0][max_index] * 100), 2)
        }
        results.append(face_data)

    return {
        "status": "success", 
        "message": f"Successfully processed {len(faces)} face(s).",
        "faces": results
    }

# --- CLI Test Block ---
if __name__ == "__main__":
    # Test the script by running `python detect_predict.py` in your terminal
    test_image = "../../data/samples/fear.jpg"
    print(f"Testing prediction on {test_image}...")
    
    result = analyze_emotion(test_image)
    
    # Print the result nicely
    import json
    print(json.dumps(result, indent=4))