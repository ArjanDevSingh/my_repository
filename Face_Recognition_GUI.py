import cv2 as cv
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

class window(QMainWindow):
    def __init__(self):
        super(window,self).__init__()
        self.setWindowTitle('Face Recognition')
        self.setGeometry(450,150,900,700)
        self.addButtons()

    def addButtons(self):
        self.existing=QPushButton('Existing',self)
        self.new=QPushButton('New',self)
        self.existing.setGeometry(350,550,100,100)
        self.new.setGeometry(460,550,100,100)
        self.existing.clicked.connect(lambda x:self.faceRecg('Training_Images'))
        self.new.clicked.connect(self.enterName)
        self.show()

    def enterName(self):
        self.name,ok=QInputDialog.getText(self,'Enter your Name','Name')
        if ok:
            self.capImg(self.name)

    def capImg(self,dirName):
        # dirName -> Name of folder in which images will be stored
        # Importing Haarcascade trained model for face
        faceCascade=cv.CascadeClassifier("haar_cascade/haarcascade_frontalface_default.xml")
        fileName=0
        seconds=5
        # Creating folder if not created yet. Will be used for storing captured images
        if dirName not in os.listdir("D:\ma folder\OPENCV\Training_Images"):
            os.mkdir('D:\ma folder\OPENCV\Training_Images/%s' %(dirName))
        capture=cv.VideoCapture(0)
        isTrue,frame=capture.read()
        while seconds:
            grayFrame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            # Detecting face in a frame
            face_coord=faceCascade.detectMultiScale(grayFrame,1.3,5,minSize=(20,20))
            # Marking detected face
            for (a,b,c,d) in face_coord:
                frame=cv.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),4)
                frame=cv.putText(frame,'Capturing Images',(a,b-15),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                # Resizing and storing the captured image in the folder created
                grayFrame=cv.resize(grayFrame[a:a+c,b:b+d],(200,200))
                cv.imwrite('Training_Images/{}/{}.jpg'.format(dirName,str(fileName)),grayFrame)
                fileName+=1
            # Showing detected face
            cv.imshow("Capturing Image",frame)
            isTrue,frame=capture.read()
            if cv.waitKey(10) & 0xFF==ord('d'):
                break
        capture.release()
        cv.destroyAllWindows()



    def readImg(self,trainingFolderPath):
        # trainingFolderPath -> Path to the folder in which has each persons folder, each containing images
        imgArray,label=[],[]
        labelCount=0
        #Accessing Folder Address, Folder Names and File Names
        for dirAddress,dirName,fileName in os.walk(trainingFolderPath):
            for folderName in dirName:
                #Creating path to particular persons folder
                subjectPath=os.path.join(dirAddress,folderName)
                for fileName in os.listdir(subjectPath):
                    try:
                        # Creating path to images in a persons folder
                        filePath=os.path.join(subjectPath,fileName)
                        img=cv.imread(filePath,0)
                        #Each image associated with its label is stored as an array in the list
                        imgArray.append(np.asarray(img,dtype=np.uint8))
                        label.append(np.asarray(labelCount,dtype=np.int32))
                    except IOError:
                        print('This is an IOError')
                    except:
                        print('Unexpected Error')
                        raise
                labelCount+=1
        return [imgArray,label]



    def faceRecg(self,PATH):
        # PATH -> Path to the folder in which has each persons folder, each containing images
        names=os.listdir(PATH)
        [imgArray,label]=window.readImg(PATH)
        # For Eigenfaces: EigenFaceRecognizer_create(), Fisherface: FisherFaceRecognizer_create(), Local Binary Pattern Histograms: LBPHFaceRecognizer_create()
        faceModel=cv.face.EigenFaceRecognizer_create()
        faceModel.train(np.asarray(imgArray),np.asarray(label))
        capture=cv.VideoCapture(0)
        faceCascade=cv.CascadeClassifier("haar_cascade/haarcascade_frontalface_default.xml")
        isTrue,frame=capture.read()
        while isTrue:
            grayFrame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            faceCoord=faceCascade.detectMultiScale(grayFrame,1.3,5,minSize=(30,30))
            for (a,b,c,d) in faceCoord:
                frame=cv.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),4)
                regionOfInterest=cv.resize(grayFrame[a:a+c,b:b+d],(200,200),interpolation=cv.INTER_LINEAR)
                # Predicting and returnng Label(ex:1,2,3..) and Confidence(ex: 67.45,88.99...) in list
                parameters=faceModel.predict(regionOfInterest)
                cv.putText(frame,names[parameters[0]],(a,b-15),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            cv.imshow("Face Recognition",frame)
            isTrue,frame=capture.read()
            if cv.waitKey(10) & 0xFF==ord('d'):
                break
        capture.release()
        cv.destroyAllWindows()

app=QApplication(sys.argv)
window=window()
app.exec_()