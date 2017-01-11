#################
# Load libraries#
#################
import cv2, os
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
import pickle

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

############################################
# function to place all faces into an array#                   
############################################
def FillImageArray(imagePath):
   print("preprocessing started")
   images = []
   gezichten = []
   i =1
   image_paths = [os.path.join(imagePath, f) for f in os.listdir(imagePath)]
   totalImages = len(image_paths)
   for image_path in image_paths:
      print("image %d/%d processed - %s"  %(i,totalImages,image_path))
      #convert to grayscale
      image = Image.open(image_path).convert('L')
      image = np.array(image, 'uint8')
      #resize to 128,128 for cascade
      image = cv2.resize(image,(128,128))
      faces = faceCascade.detectMultiScale(image)
      i = i+1
      for (x, y, w, h) in faces:
         #get every face, resize it and put it in an array
         print("preprocessing a new face")
         temp = image[y: y + h, x: x + w]
         temp = cv2.resize(temp,(128,128))
         gezichten.append(temp)
   return gezichten


###############################################
# function to add a label to an array an array#
#  1 1 -> 1 1 A                               #
#  1 1 -> 1 1 A                               #                   
###############################################
def Labelizer( array,label ):
   print("labelling the data")
   arrayLab = np.array([label])
   imax = array.shape[0]-1
   for x in range(0, imax):
      arrayLab = np.vstack([arrayLab, label])
   array=np.concatenate((array,arrayLab), axis=1)
   return array

##################################
# Makes an 1D-array of the faces:#
#  1 2 -> 1 2 3 4                #
#  3 4                           #
##################################
def Vectorizer( listIm ):
   print("extracting feature vectors from images")
   vectors =[]
   imax = len(listIm)
   for x in range(0, imax):
      temp = listIm[x].flatten()
      vectors.append(temp)
   return vectors
      

################################################################
# classifies the faces from data with the model from modelMulti#                   
################################################################
def PredicterMulti(data, modelMulti):
   print(" \nmulti class predicting started")
   results = modelMulti.predict(data)
   return results

##############################################################
# trains the model, dataInMulti contains both data and labels#                   
##############################################################
def Trainer(dataInMulti):
   tree = RandomForestClassifier()
   modelTree = tree.fit(dataInMulti[:,0:dataInMulti.shape[1]-1], dataInMulti[:,(dataInMulti.shape[1]-1)])
   return modelTree


trainPathNietRuben = 'D:/entertainment/Pictures/trainSVM/nietRuben'
trainerNietRuben = FillImageArray(trainPathNietRuben)
d2_trainerNR = Vectorizer(trainerNietRuben)
#convert everythin to an uint8 array
d2_equalisedNR = np.uint8(d2_trainerNR)
d2_trainerNietRuben = Labelizer(d2_equalisedNR,'onbekend')

trainPathRuben = 'D:/entertainment/Pictures/trainSVM/ruben'
trainerRuben = FillImageArray(trainPathRuben)
d2_trainerR = Vectorizer(trainerRuben)
d2_equalisedR = np.uint8(d2_trainerR)
d2_trainerRuben = Labelizer(d2_equalisedR, 'bekend')
#place the known (bekend) and unknown (onbekend) data into 1 array
d2_trainerMulti = np.concatenate((d2_trainerNietRuben,d2_trainerRuben), axis=0)

forest = Trainer(d2_trainerMulti)

#testing
testPath = 'D:/entertainment/Pictures/test'
tester = FillImageArray(testPath)
d2_testOne = Vectorizer(tester)
d2_testequalised = np.uint8(d2_testOne)

resultaten = PredicterMulti(d2_testequalised, forest)
print('random  forrest----------------------------------------------------------------')
print(resultaten)
pickle.dump(forest, open('RandomForrest.model', 'wb'))
