# D-tection-d-motions-Faciales---CNN-Haar-Cascade-

Brief N°5 : CNN (TensorFlow/Keras), Haar Cascade (OpenCV), FastAPI et PostgreSQL

### Période : (Deux semaines)

#### **De** : 03 Novembre 2025

#### **à** : 14 Novembre 2025

Ce projet permettra de valider la faisabilité d’un futur produit SaaS, via :

* détecter automatiquement le visage sur une photo,
  
* prédire l’émotion correspondante,
  
* enregistrer la prédiction dans une base de données.
  

**1.Préparation et exploration des données**

* Utiliser un dataset d’émotions organisé en **dossiers** nommés par émotion (ex : **happy/, sad/, angry/, surprised/, etc.**).
* Charger et prétraiter les images avec **tf.keras.utils.image_dataset_from_directory().**
* **Bonus :** Normaliser, redimensionner et augmenter les images (rotation, zoom, flip).
* Afficher des exemples d’images et des classes.

**2. Entrainement du CNN**

Créer un CNN avec TensorFlow/Keras :

* Couches Conv2D, MaxPooling2D, Flatten, Dense, Dropout.
* Optimiseur **adam**, perte **categorical_crossentropy.**

Entrainer et sauvegarder le modèle.

Afficher les métriques ***(accuracy, loss)*** et quelques prédictions sur des images test.

**3. Détection de visages avec OpenCV et Haar Cascade**

* Charger le classifieur Haar Cascade :
  
      facecascade = cv2.CascadeClassifier('haarcascadefrontalface_default.xml')
  
* Extraire la région du visage et la redimensionner à la taille attendue par le CNN.
  

Sauvegarder un script Python **detect_and_predict.py** qui :

* Charge le modèle CNN,
* Détecte le visage avec Haar Cascade,
* Fait la prédiction,
* Affiche le résultat sur l’image (avec rectangle et label).

**4. Création de l’API FastAPI**

Route POST **/predict_emotion** :

* Reçoit un fichier image via UploadFile,
* Utilise OpenCV pour détecter le visage,
* Passe la région du visage au modèle CNN pour la prédiction,
* Retourne l’émotion prédite et le score.

Route GET **/history** :

* Affiche l’historique des prédictions enregistrées dans la base PostgreSQL.

Connexion via SQLAlchemy .

Chaque prédiction est automatiquement insérée dans la base depuis la route **/predict_emotion.**

**5. Tests unitaires & GitHub Actions**

Ajouter des tests unitaires simples :

* Vérifier que ton modèle est bien sauvegarde et peut etre recharge sans erreur
* Vérification du format de la prédiction.

Automatiser les tests avec GitHub Actions

Modalités d'évaluation

* Modèle CNN correctement entraîné et sauvegardé
  
* Détection du visage via Haar Cascade opérationnelle.
  
* API FastAPI fonctionnelle et connectée à PostgreSQL.
  
* Pipeline clair : Image → Détection → Prédiction → Enregistrement.
  
* Code organisé, commenté et versionné.
  
* Tests unitaires présents.
  

Livrables

* Notebook d’entraînement du CNN.
  
* Script detect_and_predict.py (OpenCV + CNN).
  
* API FastAPI (main.py) avec routes /predict_emotion et /history.
  
* Base PostgreSQL fonctionnelle.
  
* Tests unitaires + workflow GitHub Actions.
  
* Documentation (README.md) et requirements.txt.
  
* Projet versionné sur GitHub.-- Jira
