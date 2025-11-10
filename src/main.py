from typing import Union, Annotated
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow import keras 
# from tensorflow.keras.models import models
import cv2

app = FastAPI()

# load model CNN in memory
# model = tf.keras.model.load_model('my_model_emotion_detection.keras')

# load model Haar Cascade
name_xml_file = 'haarcascade-frontalface-default.xml'
# face_cascade = cv2.CascadeClassifier(name_xml_file)

# Class Names
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

@app.get('/')
async def read_root():
    return {"Hello" : "World"}

@app.get('/item/{item_id}')
async def read_item(item_id : int, q : Union[str, None] = None):
    return {"item_id" : item_id, "q" : q}   

"""@app.post('/files/')
async def create_file(file : Annotated [bytes | None, File()] = None):
    if not file:
        return {"message" : "No file sent"}
    else:
        return {"file_size" : len(file)}"""
    
@app.post('/predict_emotion/')
async def predict_emotion(file : UploadFile = File(...)):
    if not file:
        return {"Erreur" : "file not sent"}
    
    return {'filename' : file.filename}

    contents = await file.read()    # lire les bytes : contenu du fichier

    nparr = np.frombuffer(contents, np.uint8)
    # image = cv2.imecode(nparr
