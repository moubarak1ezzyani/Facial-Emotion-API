import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from main import predict_emotion, app, Base, get_db_session
# ==== Preparation ====
# --- Fausse DB
# SQLite : ulta-rapide
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

# Engine DB for test :
engine = create_async_engine(TEST_DB_URL)

TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False 
)

# --- remplacer la vraie DB
async def override_get_db_session():
    async with TestingSessionLocal() as session:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            yield session
        finally:
            await session.close()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)

    # utiliser la fausse DB pour les test
    app.dependacy_overrides[get_db_session] = override_get_db_session

    client = TestClient(app)

# --- Test (Act & Assert)
async def test_predict_emotion_success():
    # (Si vous n'avez pas d'image, commentez la ligne 'with'
    #  et décommentez les 2 lignes suivantes pour simuler)
    # fake_image_data = b"ceci n'est pas une vraie image"
    # files = {'file': ('test.jpg', fake_image_data, 'image/jpeg')}
    
    with open("test_images/test_happy.jpg", "rb") as f:
        files = {'file': ('test_happy.jpg', f, 'image/jpeg')}   
        response = client.post("/predict_emotion/", files=files)    # Appeler l'API
         
        assert response.status_code == 200      # Le statut doit être 200
        
        # reponse --> JSON
        data = response.json()
        
        # 1. Vérifier la structure (clés : emotion, score, saved_id)
        assert "emotion" in data
        assert "score" in data
        assert "saved_id" in data
        
        # Suppose que : 'test_happy.jpg' --> "happy"
        assert data["emotion"] == "happy" 

        # 3. Vérifier les types (robustesse)
        assert isinstance(data["score"], float)
        assert isinstance(data["saved_id"], int)

        # 4. Vérifier la validité des valeursls
        
        assert data["score"] > 0.0 and data["score"] <= 1.0
        assert data["saved_id"] > 0     # Doit être un ID positif (ex: 1)
# ==================================================
"""
    import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

# Important : Importez votre 'app' et votre 'Base' depuis main.py
# (Je suppose que votre fichier s'appelle 'main')
from main import app, Base, get_db_session

# --- 1. Préparer (Arrange) : Une fausse BDD de test ---
# Nous utilisons une BDD en mémoire (SQLite) pour les tests, c'est ultra-rapide
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Créer un moteur de BDD *pour les tests*
engine = create_async_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# --- 2. Préparer (Arrange) : Remplacer la VRAIE BDD ---

# Fonction qui remplace 'get_db_session' pendant les tests
async def override_get_db_session():
    async with TestingSessionLocal() as session:
        # Crée les tables dans la BDD de test
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            yield session
        finally:
            await session.close()
            # Supprime les tables après le test
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)

# Dit à FastAPI d'utiliser notre fausse BDD pour les tests
app.dependency_overrides[get_db_session] = override_get_db_session

# Crée le "client" pour appeler notre API
client = TestClient(app)

# --- 3. Le Test (Act & Assert) ---

def test_predict_emotion_success():
    
    # Teste le endpoint /predict_emotion avec une image valide.
    
    # 3a. Préparer (Arrange) : Ouvrir une fausse image de test
    # (Créez un dossier 'test_images' avec une image 'test_happy.jpg')
    
    # (Si vous n'avez pas d'image, commentez la ligne 'with'
    #  et décommentez les 2 lignes suivantes pour simuler)
    # fake_image_data = b"ceci n'est pas une vraie image"
    # files = {'file': ('test.jpg', fake_image_data, 'image/jpeg')}
 """   
     

 

