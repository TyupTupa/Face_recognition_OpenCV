# https://habr.com/ru/companies/otus/articles/558426/
# https://habr.com/ru/articles/547218/

import cv2
import argparse

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for x, y, width, height in faces:
        face_rec = cv2.rectangle(frame, (x, y), (x+width, y+height), color=(255, 10, 0), thickness=2)
        eyes = eye_cascade.detectMultiScale(frame[y:y+height, x:x+width], scaleFactor=1.3, minNeighbors=5)
        for eye_x, eye_y, eye_width, eye_height in eyes:
            cv2.rectangle(frame[y:y+height, x:x+width], (eye_x, eye_y), (eye_x+eye_width, eye_y+eye_height), color=(255, 255, 0), thickness=2)

    cv2.imshow('Face_recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()