import os
from dotenv import load_dotenv
import tensorflow as tf
import cv2

load_dotenv()

# --- DB
# URL : postgresql+asyncpg://<user>:<password>@<host>:<port>/<db_name> 

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

# --- Jupyter Cellls
data_dir = os.getenv("data_notebook_path")
BATCH_SIZE = 32     # samples : 16, 32, 64, 128
IMG_HEIGHT = 48     # DATASET size: 48 * 48
IMG_WIDTH = 48
IMG_SIZE=(IMG_WIDTH, IMG_HEIGHT)
NUM_CLASSES = 7     # (angry, disgust, fear, happy, neutral, sad, surprise)



# -> load model CNN in memory 
model = tf.keras.models.load_model('my_model_emotion_detection.keras')


# -> load model Haar Cascade
name_xml_file = 'haarcascade-frontalface-default.xml'
face_cascade = cv2.CascadeClassifier(name_xml_file)

# -> Class Names
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']