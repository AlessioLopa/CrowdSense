from ultralytics import YOLO
import cv2
import numpy as np
 
# Charger le modèle YOLOv8
model = YOLO('yolov8s.pt')
 
# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut
 
while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image")
        break
 
    # Effectuer l'inférence sur l'image capturée
    results = model(frame, classes=[0])  # Classe 0 = 'person'
 
    # Boucler à travers les résultats pour dessiner les annotations sur l'image
    for result in results:
        annotated_frame = result.plot()  # Créer une image annotée
       
        # Convertir l'image annotée de RGB à BGR pour OpenCV
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
 
        # Afficher l'image annotée
        cv2.imshow('Webcam', annotated_frame)
 
    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Libérer la capture et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
