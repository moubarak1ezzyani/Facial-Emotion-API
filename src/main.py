from typing import Union, Annotated
from fastapi import FastAPI, File, UploadFile, Depends
import tensorflow as tf
import numpy as np
from tensorflow import keras 
from tensorflow.keras.models import Sequential
import cv2
import os
from dotenv import load_dotenv
#__Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
import psycopg2
#__table Python
from sqlalchemy import Column, Integer, Float, String, DateTime
import datetime

app = FastAPI()

# Config Engine
# --- URL : postgresql+asyncpg://<utilisateur>:<motdepasse>@<hote>:<port>/<nom_de_la_bd> ==> <> : à éliminer

load_dotenv()   # searching : file (.env) & secrets

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASS, DB_HOST, DB_NAME]):
    print("ERREUR : des variables d'env sont manquantes")
    exit()      # l'app s'arrete si le chargement des secrets est echoue

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"


# --- create engine / moteur
engine = create_async_engine(DB_URL, echo = True)  # True : logs SQL

# --- Session  : parler à la BDD
async_session_factory = sessionmaker(
    engine, class_ = AsyncSession, expire_on_commit = False
)
Base = declarative_base()

# SQLAlchemy (au demarrage): Alembic (the best)
class PredictionHistory(Base):
    __tablename__ = "EmotionTable"
    id = Column(Integer, primary_key = True,index = True )
    timestamp = Column(
        DateTime(timezone=True), 
        server_default = func.now())
    
    filename = Column(String, index = True)
    emotion = Column(String)
    score = Column(Float)

# Dependance FastAPI - Gerer la connexion
async def get_db_session() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --- load model CNN in memory 
model = tf.keras.models.load_model('src/my_model_emotion_detection.keras')
# tf_enable_onednn_opts=0
# TF_ENABLE_ONEDNN_OPTS = 0

# --- load model Haar Cascade
name_xml_file = 'src/haarcascade-frontalface-default.xml'
face_cascade = cv2.CascadeClassifier(name_xml_file)

# --- Class Names
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

@app.get('/')
async def read_root():
    return {"Hello" : "World"}

@app.get('/item/{item_id}')
async def read_item(item_id : int, q : Union[str, None] = None):
    return {"item_id" : item_id, "q" : q}   
    
@app.post('/predict_emotion/')
async def predict_emotion(file : UploadFile = File(...),db : AsyncSession = Depends(get_db_session)):
    if not file:
        return {"Erreur" : "file not sent"}
    else:
        file_name = {'filename' : file.filename}
        print(file_name)
        contents = await file.read()    # lire les bytes : contenu du fichier

        # --- NumPy + OpenCV ==> image
        np_arr = np.frombuffer(contents, np.uint8)   # tampon de lecture | u, int, 8 : unsigned, int, 8 bits --> 2^8 = 256 combinaisons (tab 1D)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    # 0 (noir/ pas de couleurs) --> 255 (blanc) | imdecode() / imread()

        # Detection Visage 
        # --- Haar Cascade : image --> niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, 1.1, 5) # Lancement Detecteur | 1.1, 5: reduction 10%, 5 confirmations de MinNeighbors

        if len(faces) == 0:
            return {"erreur" : "Aucun visage detecte"}
        
        # Preparation Visage --> CNN
        # --- Crop / Recadrer
        (x, y, w, h) = faces[0]
        face_roi = gray_image[y:y+h, x:x+w]

        # --- Redimensionner
        resized_face = cv2.resize(face_roi, (48,48))    # l'entrainement en CNN etait en 48x48

        # --- Reshape + Scaling : (1,48,48,1) = (batch,height,width,canals)
        processed_face = np.expand_dims(resized_face, axis = -1)    # (48,48) --> (48,48,1)
        processed_face = np.expand_dims(processed_face, axis = 0)   # (48,48,1) --> (1,48,48,1)

        # --- Prediction
        prediction = model.predict(processed_face)  # Résultat estimé selon classes: [[0.05, 0.1, 0.02, 0.7, 0.03, 0.05, 0.05]]
        
        # --- Extraction de donnees 
        score =  float(np.max(prediction))
        emotion_index = int(np.argmax(prediction))
        emotion_label = class_names[emotion_index]

        # => Retour DB :
        # -- Objet Python
        nouvelle_prediction = PredictionHistory(
            filename = file.filename,
            emotion = emotion_label,
            score = score
        )
        db.add(nouvelle_prediction)     # Ajouter à la DB

        await db.commit()       # Sauvegarder dans la DB

        await db.refresh(nouvelle_prediction)    # avoir l'ID créé

        # -- json File
        return {"emotion" : emotion_label, "score" : score, "saved_id" : nouvelle_prediction.id}
    
