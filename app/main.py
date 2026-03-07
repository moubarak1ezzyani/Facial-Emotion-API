from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
import numpy as np
import cv2
import os
from sqlalchemy.ext.asyncio import AsyncSession

# Internal imports
from app.database import get_db_session
from app.models import PredictionHistory
from app.services.detect_predict import analyze_emotion

app = FastAPI()

@app.get('/')
async def read_root():
    return {"Hello": "World"}

@app.post('/predict_emotion/')
async def predict_emotion(file: UploadFile = File(...), db: AsyncSession = Depends(get_db_session)):
    if not file:
        raise HTTPException(status_code=400, detail="Fichier non envoyé")
    
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Utilisez JPG ou PNG. '{file.filename.split('.')[-1]}' n'est pas supporté."
        )

    # Use the service
    result = analyze_emotion(image)
    
    if result["status"] == "error":
        return {"erreur": result["message"]}

    # Save to DB
    nouvelle_prediction = PredictionHistory(
        filename=file.filename,
        emotion=result["emotion"],
        score=result["score"]
    )
    db.add(nouvelle_prediction)
    await db.commit()
    await db.refresh(nouvelle_prediction)

    return {
        "emotion": result["emotion"], 
        "score": result["score"], 
        "saved_id": nouvelle_prediction.id
    }

    
