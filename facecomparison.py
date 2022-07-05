import cv2
from face_recognition.api import face_distance
import numpy as np
import face_recognition

Monesh=face_recognition.load_image_file('D:\PROGRAMING\PROJECT\PYTHON PROJECT\OPEN CV\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\Monesh.jpg')
Monesh=cv2.cvtColor(Monesh,cv2.COLOR_BGR2RGB)
faceloc = face_recognition.face_locations(Monesh)[0]
encodeMonesh=face_recognition.face_encodings(Monesh)[0]
cv2.rectangle(Monesh,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

Moneshtest=face_recognition.load_image_file('D:\PROGRAMING\PROJECT\PYTHON PROJECT\OPEN CV\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\Moneshtest.jpeg')
Moneshtest=cv2.cvtColor(Moneshtest,cv2.COLOR_BGR2RGB)
faceloctest = face_recognition.face_locations(Moneshtest)[0]
encodeMoneshtest=face_recognition.face_encodings(Moneshtest)[0]
cv2.rectangle(Moneshtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeMonesh],encodeMoneshtest)
facedis=face_recognition.face_distance([encodeMonesh],encodeMoneshtest)
print(results,facedis)
cv2.putText(Moneshtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Monesh',Monesh)
cv2.imshow('Moneshtest',Moneshtest)
cv2.waitKey(0)