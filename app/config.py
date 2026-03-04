import os
from dotenv import load_dotenv
import tensorflow as tf
import cv2

# --- URL : postgresql+asyncpg://<user>:<password>@<host>:<port>/<db_name> 
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

# --- load model CNN in memory 
model = tf.keras.models.load_model('my_model_emotion_detection.keras')
# tf_enable_onednn_opts=0
# TF_ENABLE_ONEDNN_OPTS = 0

# --- load model Haar Cascade
name_xml_file = 'haarcascade-frontalface-default.xml'
face_cascade = cv2.CascadeClassifier(name_xml_file)

# --- Class Names
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']