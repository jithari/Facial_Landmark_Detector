
import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
   _, img  = cap.read()
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   faces = detector(gray)
   for face in faces:
       x1 = face.left()
       y1 = face.top()
       x2 = face.right()
       y2 = face.bottom()

       cv2.rectangle(img, (x1, y1), (x2, y2), (100, 0, 100), 3)

       landmarks = predictor(gray, face)

       for p in range(0, 68):
            x = landmarks.part(p).x
            y = landmarks.part(p).y
            cv2.circle(img, (x, y), 2, (p+10, 0, 155), -1)


   cv2.imshow("Image", img)

   key = cv2.waitKey(1)
   if key == ord('q'):
       break
