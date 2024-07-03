import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/elon1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/elon3.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgmz = face_recognition.load_image_file('ImagesBasic/mzuo.jpg')
imgmz = cv2.cvtColor(imgmz, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2)

faceLocmz = face_recognition.face_locations(imgmz)[0]
encodemz = face_recognition.face_encodings(imgmz)[0]
cv2.rectangle(imgmz, (faceLocmz[3], faceLocmz[0]), (faceLocmz[1], faceLocmz[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis1 = face_recognition.face_distance([encodeElon], encodeTest)
faceDis2 = face_recognition.face_distance([encodeElon], encodemz)
print(results)
print(faceDis1, faceDis2)

cv2.putText(imgTest, f'{results} {round(faceDis1[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)


cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.imshow('Michael Z', imgmz)
cv2.waitKey(0)