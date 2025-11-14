import cv2
import os

# --- Definir les chemins Machine Learning\haarcascade-frontalface-default.xml
cascade_path = "src/haarcascade-frontalface-default.xml"     # modele XML
image_path = "./Data_Kaggle_Emotional_Detection/train/happy/im7.png"      # image à detecter
output_path = "result_image.jpg"    # output file Name

# --- verifier si les fichiers existent 
if not os.path.exists(cascade_path):
    print(f"Erreur : fichier cascade introuvable dans {cascade_path}")
    exit()

if not os.path.exists(image_path):
    print(f"Erreur : image introuvable dans {image_path}")
    exit()

# --- loading Classifier & image
print("Chargement du modele Haar Cascade...")
face_cascade = cv2.CascadeClassifier(cascade_path)      # Creer detecteur & Charger XML

print("Chargement d'image...")
image = cv2.imread(image_path)

# --- Pretraitement (Crucial) ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- lancer la detection
print("Detection de Visage...")

# Faces : liste de rectangles (x, y, Width, Height)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # À quel point réduire l'image à chaque "passage"
    minNeighbors=5,   # Combien de "voisins" chaque rectangle doit avoir
    minSize=(30, 30)  # La taille minimale d'un visage à détecter
)

print(f"Trouvé : {len(faces)} visage(s) !")

# --- Dessiner les rectangles ---
# Boucle sur chaque visage/rectangle trouvé 
for (x, y, w, h) in faces:
   cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
   """ Dessine un rectangle sur l'image ORIGINALE (couleur)
    (x, y) = coin supérieur gauche
    (x+w, y+h) = coin inférieur droit
    (0, 255, 0) = couleur (Red, Green, Blue) 
    2 = épaisseur de la ligne
    """ 
# --- Sauvegarder le résultat ---
cv2.imwrite(output_path, image)
print(f"Image sauvegardée sous : {output_path}")

# image --> fenêtre
cv2.imshow('Visages trouvés', image)
cv2.waitKey(0)      # Attend qu'on appuie sur une touche
cv2.destroyAllWindows()

# imdecode()  /  imread()  /   imwrite()  / imshow()