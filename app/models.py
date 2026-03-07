from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func 
from app.database import Base

class PredictionHistory(Base):
    __tablename__ = "EmotionTable"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    emotion = Column(String)
    
    # 'score' to 'confidence' 
    confidence = Column(Float) 
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())