# 😐 Facial Emotion API

## 📄 Contexte du Projet

Ce projet est un prototype d'API REST développé pour une startup d'analyse UX. Il combine **Vision par Ordinateur** et **Deep Learning** pour analyser les émotions faciales en temps réel et stocker les résultats pour des études statistiques.

L'application expose une API performante (FastAPI asynchrone) capable de détecter un visage, classifier son émotion parmi 7 catégories (*Happy, Sad, Angry, Surprise, Neutral, Fear, Disgusted*) et archiver la donnée.

## ⚙️ Architecture Technique

Le pipeline de traitement suit ces étapes rigoureuses :

1. **Réception** : L'API reçoit une image via l'endpoint `/predict_emotion`.
2. **Détection (OpenCV)** : Le classifieur *Haar Cascade* isole le visage.
3. **Normalisation** : Recadrage, conversion en niveaux de gris, redimensionnement (48x48px) et mise à l'échelle [0-1].
4. **Inférence (CNN)** : Le modèle TensorFlow (`.keras`) prédit l'émotion et le score de confiance.
5. **Persistance (SQLAlchemy Async)** : Enregistrement non-bloquant dans PostgreSQL.

---

## 📂 Structure du Projet

```bash
FACIAL-EMOTION-API/
├── .github/                             # Reste tel quel (pour tes futures Actions)
├── Data_Kaggle_Emotional_Detection/     # Reste tel quel (ignoré par Git)
├── models/                              # 🆕 NOUVEAU : Pour tes poids d'IA
│   ├── haarcascade-frontalface-default.xml
│   └── my_model_emotion_detection.keras
├── notebooks/                           # 🆕 NOUVEAU : Pour ton exploration
│   └── eda.ipynb
|   └── training.ipynb
|   └── detection_predict.ipynb                         
├── app/                                 # 🔄 TON ANCIEN 'src' (dédié à l'API)
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── create_tables.py
│   └── main.py
├── tests/                               # 🆕 NOUVEAU : Pour tes tests unitaires
│   └── test_unitaire.py                 
├── detect_and_predict.py                # 🔄 Script autonome pour la détection et visualisation
├── fear.jpg                             # (Image de test, tu peux la laisser là ou créer un dossier 'samples')
├── venv/
├── .env
├── .gitignore
├── Notes.md
└── README.md
```

---

## 🚀 Installation et Configuration

### 1. Pré-requis

* Python 3.9+
* PostgreSQL installé et service actif.

### 2. Installation

Cloner le dépôt et installer les librairies :

```bash
git clone https://github.com/votre-user/Facial-Emotion-API.git
pip install -r requirements.txt

```

### 3. Configuration de la Base de Données

Créez un fichier `.env` à la racine du projet pour vos variables d'environnement (sécurité) :

```env
DB_USER=postgres
DB_PASS=votre_mot_de_passe
DB_HOST=localhost
DB_NAME=emotion_db

```

Initialisez les tables dans la base de données avec le script dédié :

```bash
python create_tables.py

```

*(Cela créera la table `EmotionTable` via SQLAlchemy).*

---

## 💻 Utilisation

### Lancer l'API (Serveur)

Démarrer le serveur Uvicorn avec rechargement automatique :

```bash
uvicorn main:app --reload

```

L'API sera accessible sur : `http://127.0.0.1:8000`

### Tester avec le script autonome

Si vous souhaitez tester la détection et la prédiction sur une image locale sans passer par le serveur :

```bash
python detect_and_predict.py

```

*(Assurez-vous de modifier le chemin `image_path` dans le fichier avant).*

---

## 📡 Documentation des Endpoints

Une documentation interactive (Swagger UI) est disponible automatiquement sur `http://127.0.0.1:8000/docs`.

### 1️⃣ `POST /predict_emotion`

Analyse une image envoyée par l'utilisateur.

* **Input** : Fichier image (`UploadFile`).
* **Processus** : Détection -> Prédiction -> Sauvegarde DB.
* **Output (JSON)** :
```json
{
  "face_detected": true,
  "emotion": "happy",
  "confidence": 0.98,
  "processing_time": "0.04s"
}

```



### 2️⃣ `GET /history`

Récupère l'historique des analyses stockées en base.

* **Output** : Liste des entrées (ID, Emotion, Confiance, Date).

---

## ✅ Qualité du Code & Tests

Le projet intègre des tests unitaires pour garantir la robustesse du modèle.

**Exécuter les tests :**

```bash
pytest test_unitaire.py

```

**Couverture des tests :**

* `test_model_save_and_load` : Vérifie l'intégrité de la sauvegarde/chargement du modèle `.keras`.
* `test_prediction_format` : Vérifie que le modèle renvoie bien un tenseur de forme `(1, 7)`.

---

## 🧠 Détails du Modèle (CNN)

* **Entraînement** : Notebook `notebooks/training.ipynb`.
* **Input** : Images 48x48 pixels, Grayscale (1 canal).
* **Classes (7)** : `Angry`, `Disgusted`, `Fearful`, `Happy`, `Neutral`, `Sad`, `Surprised`.
* **Performance** : Modèle optimisé pour la rapidité d'inférence (convient au temps réel).

## branchs
feature/1-cnn-training

Ce que tu y fais : L'étape 1 et 2 du brief. Préparation du dataset, création du notebook, entraînement du modèle TensorFlow/Keras, et sauvegarde du fichier .keras.

feature/2-opencv-detection

Ce que tu y fais : L'étape 3 du brief. Création du script detect_and_predict.py qui charge le Haar Cascade, détecte le visage, et utilise ton modèle CNN pour afficher la prédiction sur l'image.

feature/3-fastapi-postgres

Ce que tu y fais : L'étape 4 du brief. C'est le gros morceau. Création de l'architecture de ton API (app/main.py), connexion à la base de données PostgreSQL via SQLAlchemy, et création des routes /predict_emotion (POST) et /history (GET).

feature/4-tests-and-ci

Ce que tu y fais : L'étape 5 du brief. Rédaction de tes tests unitaires avec Pytest (tests/test_unitaire.py) et création du dossier .github/workflows/ pour tes GitHub Actions.

docs/readme-et-livrables

Ce que tu y fais : La touche finale. Rédaction de ton README.md (documentation), mise au propre de ton requirements.txt et vérification de ton .gitignore.

## 📂 Recommended Naming Convention
StepFile NamePurpose  `0101_eda_exploration.ipynb` Data visualization, class distribution, and image checks.`0202_cnn_training.ipynb` Model architecture, data augmentation, and training logic.`0303_detection_inference.ipynb` Testing the model on new images and visualizing predictions.