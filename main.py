import cv2

#pretrained models (Haar cascades) for face and eye reсognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#capture camera with index 0
cap = cv2.VideoCapture(0)

#while the programm is working
while True:
    ret, frame = cap.read()

    #searches for faces on a frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for x, y, width, height in faces:

        #draws rectangle around each face
        face_rec = cv2.rectangle(frame, (x, y), (x+width, y+height), color=(0, 10, 100), thickness=2)

        #finds eyes on each face
        eyes = eye_cascade.detectMultiScale(frame[y:y+height, x:x+width], scaleFactor=1.3, minNeighbors=5)

        #draws rectangles around each eye in current face
        for eye_x, eye_y, eye_width, eye_height in eyes:
            cv2.rectangle(frame[y:y+height, x:x+width], (eye_x, eye_y), (eye_x+eye_width, eye_y+eye_height), color=(255, 255, 0), thickness=2)

    #shows the result on the video stream
    cv2.imshow('Face_recognition', frame)

    #exit conditions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()