# Détection d’Émotions Faciales (CNN + FastAPI + PostgreSQL)

## 1\. Vue d'ensemble du Projet

Ce projet est un prototype d'API RESTful pour l'analyse de sentiments en temps réel via la Vision par Ordinateur. L'application est capable de recevoir une image, d'y détecter un visage, de prédire l'émotion faciale correspondante à l'aide d'un réseau de neurones convolutif (CNN), et d'historiser ce résultat dans une base de données PostgreSQL pour une analyse ultérieure.

Ce prototype sert de "Preuve de Concept" (PoC) pour des cas d'usage avancés, tels que l'analyse de l'expérience utilisateur (UX) ou l'étude de réactions à des produits.

### Fonctionnalités Clés

  * **Endpoint `POST /predict_emotion/` :** Accepte un upload d'image, effectue la détection et la prédiction, et sauvegarde le résultat.
  * **Endpoint `GET /history/` :** (Si implémenté) Retourne un historique complet de toutes les prédictions stockées.
  * **Intégration BDD :** Connexion asynchrone à PostgreSQL via SQLAlchemy pour une persistance des données non bloquante.
  * **CI/CD :** Pipeline d'intégration continue avec GitHub Actions pour automatiser l'exécution des tests unitaires à chaque `push`.

-----

## 2\. Architecture et Démarche

L'application suit un pipeline de traitement clair, de la réception de la requête à la sauvegarde des données.

### Démarche (Pipeline de la Requête)

Le flux de données pour une prédiction est le suivant :

1.  **Client** envoie une image (`POST /predict_emotion/`).
2.  **FastAPI** reçoit le fichier et obtient une session de la BDD (via `Depends`).
3.  **OpenCV** décode l'image et le modèle **Haar Cascade** détecte les coordonnées du visage.
4.  Le visage est découpé, redimensionné (48x48, gris) et préparé pour le modèle.
5.  **TensorFlow (CNN)** prédit le vecteur de probabilité des 7 émotions.
6.  Le score le plus élevé et l'émotion correspondante sont extraits.
7.  **SQLAlchemy** utilise la session pour insérer la prédiction dans la table PostgreSQL (`await db.commit()`).
8.  **FastAPI** retourne la réponse JSON (`{"emotion": ..., "score": ...}`).

### Arborescence du Projet

La structure du dépôt est organisée pour séparer la configuration, le code source et les tests.

```
.
├── .github/workflows/
│   └── python-ci.yml       # Workflow d'intégration continue (Tests)
├── src/
│   ├── __init__.py
│   ├── main.py             # Logique API (FastAPI), modèles BDD, endpoints
│   ├── test_unitaire.py    # Tests unitaires (Pytest)
│   │
│   ├── my_model_emotion_detection.keras  # Modèle CNN entraîné
│   └── haarcascade_frontalface_default.xml # Modèle OpenCV
│
├── .env                    # Fichier de secrets (LOCAL - ignoré par Git)
├── .env.example            # Template pour les variables d'environnement
├── .gitignore
├── requirements.txt        # Dépendances Python
└── README.md               # Cette documentation
```

-----

## 3\. Stack Technique (Technologies et Justifications)

| Technologie | Utilisation | Raison du Choix |
| :--- | :--- | :--- |
| **Python 3.11** | Langage principal. | Écosystème mature pour l'IA et le développement web. |
| **FastAPI** | Framework de l'API REST. | **Asynchrone natif** (haute performance pour les E/S), auto-documentation (Swagger UI), validation de données (Pydantic). |
| **TensorFlow (Keras)** | Entraînement et prédiction (CNN). | Écosystème complet pour le Deep Learning ; l'API Keras facilite le prototypage rapide de modèles. |
| **OpenCV (Haar Cascade)** | Détection de visage. | **Légèreté et rapidité**. Extrêmement efficace pour la *détection* (trouver le visage), même sur des CPU. |
| **PostgreSQL** | Base de données relationnelle. | **Robustesse et scalabilité**. Solution "standard" pour les applications de production nécessitant des écritures fiables. |
| **SQLAlchemy (AsyncPG)** | ORM (Pont BDD) et Pilote. | SQLAlchemy est l'ORM standard en Python. `AsyncPG` est le pilote asynchrone le plus rapide pour PostgreSQL, s'intégrant parfaitement avec FastAPI. |
| **Pytest** | Framework de tests unitaires. | Syntaxe simple (`assert`), gestion puissante des "fixtures", et standard de l'industrie pour les tests Python. |
| **GitHub Actions** | Intégration Continue (CI/CD). | Automatisation des tests directement depuis le dépôt Git, assurant la non-régression du code sur la branche `main`. |
| **Python-dotenv** | Gestion des secrets. | Meilleure pratique de sécurité pour charger les configurations (identifiants BDD) depuis un fichier `.env` plutôt que de les coder en dur. |

-----

## 4\. Guide d'Installation et d'Exécution

### Prérequis

  * Python (version 3.10+ recommandée)
  * Un serveur PostgreSQL en cours d'exécution (localement ou distant)
  * Git

### Étape 1 : Cloner le Dépôt

```bash
git clone [URL_HTTPS_DU_DEPOT]
cd [NOM_DU_DEPOT]
```

### Étape 2 : Configurer l'Environnement Virtuel

Il est crucial d'isoler les dépendances du projet.

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Windows (PowerShell/CMD)
.\venv\Scripts\activate
# Sur Mac/Linux
# source venv/bin/activate
```

### Étape 3 : Installer les Dépendances

```bash
# Mettre à jour pip et installer les paquets
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 4 : Configurer la Base de Données et les Secrets

1.  **Base de Données :**
    Assurez-vous que votre serveur PostgreSQL est actif. Connectez-vous (via `psql` ou `pgAdmin`) et créez une nouvelle base de données.

    ```sql
    CREATE DATABASE emotion_db;
    ```

2.  **Variables d'Environnement :**
    Créez un fichier nommé `.env` à la racine du projet (utilisez `.env.example` comme modèle) et remplissez-le avec vos identifiants.

    ```ini
    # Fichier: .env
    DB_USER=postgres
    DB_PASS=votre_mot_de_passe_secret
    DB_HOST=localhost
    DB_NAME=emotion_db
    ```

### Étape 5 : Lancer l'Application

Le serveur FastAPI est configuré avec un événement `lifespan`. Au démarrage, il se connectera à la base de données et exécutera `Base.metadata.create_all` pour créer automatiquement la table `EmotionTable` si elle n'existe pas.

```bash
# (En supposant que main.py est dans le dossier 'src')
uvicorn src.main:app --reload
```

L'API est maintenant en cours d'exécution sur `http://127.0.0.1:8000`.

-----

## 5\. Documentation des Endpoints

L'interface de test interactive (Swagger UI) est disponible à l'adresse :
**`http://127.0.0.1:8000/docs`**

### Endpoint Racine

  * **`GET /`**
      * **Description :** Endpoint de "health check" pour vérifier que l'API est en ligne.
      * **Réponse :** `{"Hello": "World"}`

### Endpoint de Prédiction

  * **`POST /predict_emotion/`**
      * **Description :** Reçoit une image, analyse l'émotion du visage principal, et sauvegarde le résultat.
      * **Input (Corps) :** `file` (UploadFile). Formats supportés : JPG, PNG.
      * **Réponse (Succès 200 OK) :**
        ```json
        {
          "emotion": "happy",
          "score": 0.954,
          "saved_id": 42
        }
        ```
      * **Réponses (Erreurs) :**
          * `400 Bad Request` : Si le format de l'image n'est pas supporté (ex: AVIF, WEBP).
          * `500 Internal Server Error` : Si aucun visage n'est détecté ou si une autre erreur de traitement survient.

### Endpoint d'Historique

  * **`GET /history/`**
      * **Description :** (Implémentation requise) Doit retourner la liste de toutes les prédictions stockées dans la table `EmotionTable`.

-----

## 6\. Tests et Intégration Continue

### Tests Unitaires

Le projet utilise `pytest` pour les tests unitaires et d'intégration. Les tests sont configurés pour utiliser une base de données **SQLite en mémoire** (`aiosqlite`) afin d'isoler les tests de la base de données de développement (PostgreSQL).

Pour exécuter les tests localement :

```bash
pytest
```

### Intégration Continue (GitHub Actions)

Le workflow défini dans `.github/workflows/python-ci.yml` s'exécute automatiquement à chaque `push` ou `pull request` vers la branche `main`.

Le pipeline CI effectue les étapes suivantes :

1.  Met en place une machine virtuelle Ubuntu.
2.  Installe Python 3.11.
3.  Installe toutes les dépendances listées dans `requirements.txt`.
4.  Exécute `pytest`. (Note : des variables d'environnement factices sont fournies au workflow pour permettre à `main.py` de s'importer sans erreur).
