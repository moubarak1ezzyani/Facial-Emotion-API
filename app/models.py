from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base

class PredictionHistory(Base):
    __tablename__ = "EmotionTable"
    id = Column(Integer, primary_key = True,index = True )
    """ timestamp = Column(
        DateTime(timezone=True), 
        server_default = func.now()) """
    
    filename = Column(String, index = True)
    emotion = Column(String)
    score = Column(Float)