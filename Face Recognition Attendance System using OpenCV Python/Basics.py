import cv2
import numpy as np
import face_recognition


imgRaihan = face_recognition.load_image_file('ImagesBasic/Raihan Sikdar.jpg')
imgRaihan = cv2.cvtColor(imgRaihan, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Zarin Tasnim.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgRaihan = face_recognition.load_image_file('ImagesBasic/Kazi Sarwar.jpg')
imgRaihan = cv2.cvtColor(imgRaihan, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Zarin Tasnim.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgRaihan = face_recognition.load_image_file('ImagesBasic/Raihan Sikdar.jpg')
imgRaihan = cv2.cvtColor(imgRaihan, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Kazi Sarwar.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


imgRaihan = face_recognition.load_image_file('ImagesBasic/Raihan Sikdar.jpg')
imgRaihan = cv2.cvtColor(imgRaihan, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Raihan.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgRaihan = face_recognition.load_image_file('ImagesBasic/Kazi Sarwar.jpg')
imgRaihan = cv2.cvtColor(imgRaihan, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Kazi Sarwar2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)



faceLoc = face_recognition.face_locations(imgRaihan)[0]
encodeRaihan = face_recognition.face_encodings(imgRaihan)[0]
cv2.rectangle(imgRaihan, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeRaihan], encodeTest)
faceDis = face_recognition.face_distance([encodeRaihan], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Main Images', imgRaihan)
cv2.imshow('Test Images', imgTest)
cv2.waitKey(0)