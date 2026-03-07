import pytest
import tensorflow as tf
import numpy as np
import os
from app.services.detect_predict import emotion_model as model 

def test_model_save_and_load(tmp_path):
    """
    Vérifie que le modèle peut être sauvegardé et rechargé sans erreur.
    """
    # --- Préparer (Arrange)
    # tmp_path est un outil magique de Pytest qui crée un dossier temporaire
    # et le supprime automatiquement à la fin du test !
    temp_model_path = tmp_path / "test_model.keras"
    
    # --- Action
    model.save(temp_model_path)
    loaded_model = tf.keras.models.load_model(temp_model_path)
    
    # --- Vérifier (Assert)
    assert temp_model_path.exists()      # Vérifie que le fichier a bien été créé
    assert loaded_model is not None      # Vérifie que le modèle rechargé n'est pas vide
    
    # Plus besoin de os.remove() ! Pytest nettoie tout seul.

def test_prediction_format():
    """
    Vérifie que le format de la sortie de prédiction est (1, 7).
    """
    # 1. Préparer (Arrange)
    # Crée une fausse image (1 lot, 48x48 pixels, 1 canal gris)
    # On utilise random.rand pour simuler des pixels normalisés (entre 0 et 1)
    dummy_input = np.random.rand(1, 48, 48, 1)
    
    # 2. Agir (Act)
    prediction = model.predict(dummy_input, verbose=0)
    
    # 3. Vérifier (Assert)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 7)    # 1 image, 7 émotions possibles