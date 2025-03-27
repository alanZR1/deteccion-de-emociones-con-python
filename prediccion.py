# realtime_emotion_detection.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('deteccion_gestos.h5')

# Etiquetas de emociones
emotion_labels = {
    0: 'Enojo',
    1: 'Disgusto',
    2: 'Miedo',
    3: 'Felicidad',
    4: 'Neutral',
    5: 'Tristeza',
    6: 'Sorpresa'
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roi_gray = gray[y: y + h, x: x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        
        # Normalizar la imagen
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis = 0)
        roi = np.expand_dims(roi, axis = -1)
        
        # Predecir la emoción
        prediction = model.predict(roi)[0]
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = np.max(prediction)
        
        # Mostrar la emoción y confianza
        text = f"{emotion} ({confidence * 100: .1f } % )"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Mostrar el frame resultante
    cv2.imshow('Detección de Emociones', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()