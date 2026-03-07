import cv2
import numpy as np
import tensorflow as tf
import os

# Paths are simple because this script runs from the root folder!
CASCADE_PATH = 'models/haarcascade-frontalface-default.xml'
MODEL_PATH = 'models/my_model_emotion_detection_1.keras'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_model = tf.keras.models.load_model(MODEL_PATH)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

def detect_and_draw(image_path, output_path="output/result_visual.jpg"):
    print(f"Processing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found. Please check the path.")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
        
        prediction = emotion_model.predict(roi_reshaped, verbose=0)
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        confidence = float(np.max(prediction)) * 100
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 8)
        text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✅ Done! Image successfully saved to: {output_path}")

if __name__ == "__main__":
    # Point this to a real image in your data folder
    test_image = "data/samples/sad.jpg" 
    detect_and_draw(test_image)