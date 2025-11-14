import pytest
import tensorflow as tf
import numpy as np
import os
from main import model 

def test_model_save_and_load():
    """
    Vérifie que le modèle peut être sauvegardé et rechargé sans erreur.
    """
    # --- Préparer (Arrange)
    # chemin temporaire : fichier de test
    temp_model_path = "my_model_emotion_detection.keras"
    
    # --- Action
    model.save(temp_model_path)     # Sauvegarde le modèle et le recharge immédiatement
    loaded_model = tf.keras.models.load_model(temp_model_path)
    
    # --- assert
    assert os.path.exists(temp_model_path)      # Verfication : le fichier a bien ete cree
    assert loaded_model is not None         # Verificcation : le mdoele rechargé n'est pas vide
    
 
    os.remove(temp_model_path)         # Cleaning

def test_prediction_format():
    """
    Vérifie que le format de la sortie de prédiction est (1, 7).
    """
    # 1. Préparer (Arrange)
    # Crée une fausse image (1 lot, 48x48 pixels, 1 canal gris)
    # C'est la forme que votre modèle attend en entrée.
    dummy_input = np.ones((1, 48, 48, 1))
    
    # 2. Agir (Act)
    # Exécute la prédiction
    prediction = model.predict(dummy_input)
    
    # 3. Vérifier (Assert)
    # Vérifie que la sortie est bien un array Numpy
    assert isinstance(prediction, np.ndarray)
    
    # Vérifie que la forme est (1, 7) :
    # 1 image dans le lot, 7 classes (émotions) en sortie.
    assert prediction.shape == (1, 7)