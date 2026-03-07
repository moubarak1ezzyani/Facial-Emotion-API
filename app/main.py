import os
import shutil
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

# Import your database functions and models (adjust these imports based on your actual files)
from app.database import get_db_session
from app.models import PredictionHistory # Assuming you have this defined somewhere
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
        # This automatically handles normalization, multiple faces, and model loading!
        result = analyze_emotion(temp_file_path)

        if result["status"] == "error":
            return {"erreur": result["message"]}
        
        if len(result["faces"]) == 0:
            return {"erreur": "Aucun visage detecte"}

        # 4. Save the FIRST detected face to the database (to match your original logic)
        first_face = result["faces"][0]
        
        nouvelle_prediction = PredictionHistory(
            filename=file.filename,
            emotion=first_face["emotion"],
            score=first_face["confidence"]
        )
        db.add(nouvelle_prediction)
        await db.commit()
        await db.refresh(nouvelle_prediction)

        # 5. Return JSON to user
        return {
            "emotion": first_face["emotion"], 
            "score": first_face["confidence"], 
            "saved_id": nouvelle_prediction.id,
            "all_faces_detected": result["faces"] # Bonus: Give them the bounding boxes too!
        }
        
    finally:
        # 6. Always clean up the temp image
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
