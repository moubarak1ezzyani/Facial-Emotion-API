import os
from dotenv import load_dotenv

# --- Dynamically find the project root (FACIAL-EMOTION-API) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

# --- DB Configuration ---
DB_USER = (os.getenv("DB_USER") or "").strip()
DB_PASS = (os.getenv("DB_PASS") or "").strip()
DB_HOST = (os.getenv("DB_HOST") or "").strip()
DB_NAME = (os.getenv("DB_NAME") or "").strip()

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

# --- Paths ---
# Use absolute paths based on BASE_DIR to prevent "File not found" errors
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "my_model_emotion_detection.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade-frontalface-default.xml")

# --- Class Names ---
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']