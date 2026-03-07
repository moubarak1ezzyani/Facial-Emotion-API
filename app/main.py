import os
import shutil
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select 

# Import your database functions and models
from app.database import get_db_session
from app.models import PredictionHistory 
from app.config import DB_HOST, DB_PASS, DB_NAME, DB_USER, DB_URL

# Import our working ML Service!
from app.services.detect_predict import analyze_emotion

app = FastAPI()

# Config Engine Check
if not all([DB_USER, DB_PASS, DB_HOST, DB_NAME]):
    print("ERREUR : des variables d'env sont manquantes")
    exit()

@app.get('/')
async def read_root():
    return {"Hello": "World"}

# --- NEW ROUTE : GET HISTORY ---
@app.get('/history/')
async def get_history(db: AsyncSession = Depends(get_db_session)):
    """
    Récupère l'historique complet des prédictions depuis PostgreSQL.
    """
    # Récupère toutes les prédictions, triées par date (les plus récentes en premier)
    result = await db.execute(select(PredictionHistory).order_by(PredictionHistory.created_at.desc()))
    history = result.scalars().all()
    return history

# --- UPDATED ROUTE : POST PREDICTION ---
@app.post('/predict_emotion/')
async def predict_emotion(file: UploadFile = File(...), db: AsyncSession = Depends(get_db_session)):
    # 1. Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Utilisez JPG ou PNG. '{file.filename}' n'est pas supporté.")

    # 2. Save the uploaded file temporarily to disk
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. Use the working service to do all the heavy AI lifting
        result = analyze_emotion(temp_file_path)

        if result["status"] == "error":
            return {"erreur": result["message"]}
        
        if len(result["faces"]) == 0:
            return {"erreur": "Aucun visage detecte"}

        # 4. Save the FIRST detected face to the database
        first_face = result["faces"][0]
        
        nouvelle_prediction = PredictionHistory(
            filename=file.filename,
            emotion=first_face["emotion"],
            confidence=first_face["confidence"] # <-- UPDATED: Changed 'score' to 'confidence'
        )
        db.add(nouvelle_prediction)
        await db.commit()
        await db.refresh(nouvelle_prediction)

        # 5. Return JSON to user
        return {
            "emotion": first_face["emotion"], 
            "confidence": first_face["confidence"], # <-- UPDATED: Changed 'score' to 'confidence'
            "saved_id": nouvelle_prediction.id,
            "all_faces_detected": result["faces"]
        }
        
    finally:
        # 6. Always clean up the temp image
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)