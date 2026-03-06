import os
from dotenv import load_dotenv

load_dotenv()

# --- Dynamically find the project root (FACIAL-EMOTION-API) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- DB Configuration ---
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

# --- Jupyter / Data Paths ---
data_dir = os.getenv("data_notebook_path")

# Note: Using absolute paths based on BASE_DIR prevents "File not found" errors
TRAIN_DIR = os.path.join(BASE_DIR, "data","Data_Kaggle_Emotional_Detection", "train")
TEST_DIR = os.path.join(BASE_DIR, "data","Data_Kaggle_Emotional_Detection", "test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "my_model_emotion_detection_2.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade-frontalface-default.xml")

# --- ML Hyperparameters ---
BATCH_SIZE = 64     # samples : 16, 32, 64, 128
IMG_HEIGHT = 48     # DATASET size: 48 * 48
IMG_WIDTH = 48
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
NUM_CLASSES = 7     # (angry, disgust, fear, happy, neutral, sad, surprise)
EPOCHS = 25

# -> Class Names
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']