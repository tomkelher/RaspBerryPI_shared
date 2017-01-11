#################
# Load libraries#
#################
import sys
import cv2, os
import serial
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from PIL import Image
from picamera.array import PiRGBArray
from picamera import PiCamera
import pickle


def PredicterMulti(data, modelMulti):
   results = modelMulti.predict(data)
   return results

#setup the port for serial connection
port = serial.Serial("/dev/ttyAMA0",baudrate=115200,timeout=3.0)

#load in the cascade
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
gezicht = []

#setup the Raspberry Pi cam
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = (320,240))

time.sleep(0.1)

#run this code for each frame
for frame in camera.capture_continuous(rawCapture, format ="bgr", use_video_port = True):
    imageShow = frame.array
    imageUse = cv2.cvtColor(imageShow, cv2.COLOR_BGR2GRAY)
    imageUse = np.array(imageUse, 'uint8')
    imageUse = cv2.resize(imageUse,(128,128))
    faces = faceCascade.detectMultiScale(imageUse,scaleFactor=1.1,minNeighbors=5,minSize=(15, 15),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        temp = imageUse[y: y + h, x: x + w]
        temp = cv2.resize(temp,(128,128))
        gezicht = temp.flatten()
    cv2.imshow('Face_recognise_application',imageShow)
    #space button
    a = cv2.waitKey(32) & 0xff
    #escape button
    k = cv2.waitKey(27) & 0xff
    rawCapture.truncate(0)
    #when space is pressed, terminate cam (32 = ascii for space)
    if a == 32:
        break
    #when esc is pressed, terminate cam and clear faces (32 = ascii for esc)
    if k == 27:
        gezicht = []
        break

# When everything is done, release the capture
cv2.destroyAllWindows()

if (len(gezicht) == 0):
    message = 'image recognition terminated/nno face is being processed'
else:
    d2_gezicht = np.uint8(gezicht)
    model = pickle.load(open('RandomForrest.model', 'rb'))
    detectie = PredicterMulti(d2_gezicht, model)
    message = detectie.tostring()

print(message)
port.isOpen()
port.write(message)
    
