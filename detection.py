from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle YOLOv8
model = YOLO('yolov8s.pt')

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

# Initialiser les variables
active_detections = set()  # Ensemble pour suivre les personnes détectées

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image")
        break

    # Effectuer l'inférence sur l'image capturée
    results = model(frame, classes=[0])  # Classe 0 = 'person'

    current_detections = set()  # Ensemble pour les détections actuelles

    # Boucler à travers les résultats pour dessiner les annotations sur l'image
    for result in results:
        for box in result.boxes:  # Parcourir chaque boîte détectée
            # Extraire les coordonnées de la boîte
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_detections.add((x1, y1, x2, y2))  # Ajouter la boîte aux détections actuelles

            # Dessiner la boîte sur l'image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mettre à jour les détections actives
    active_detections = current_detections

    # Compter le nombre de personnes détectées
    person_count = len(active_detections)

    # Déterminer la densité et préparer le texte
    if person_count <= 10:
        density_text = 'Densite: Faible'
    elif 10 < person_count <= 20:
        density_text = 'Densite: Moderee'
    elif 20 < person_count <= 30:
        density_text = 'Densite: Moyenne'
    else:
        density_text = 'Densite: Elevee'

    # Afficher le compteur de personnes et la densité sur l'image
    #cv2.putText(frame, f'Person Count: {person_count}', (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, density_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher l'image annotée
    cv2.imshow('Webcam', frame)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
