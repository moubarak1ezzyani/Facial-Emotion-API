import asyncio
from dotenv import load_dotenv
import os

from main import Base, PredictionHistory, engine 

async def init_db():
    print("Connexion à la base de données...")
    async with engine.begin() as conn:
        print("Suppression des anciennes tables (si elles existent)...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Création de la nouvelle table 'EmotionTable'...")
        await conn.run_sync(Base.metadata.create_all)
    print("Table 'EmotionTable' créée avec succès.")

if __name__ == "__main__":
    asyncio.run(init_db())