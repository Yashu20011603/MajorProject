import cv2
import numpy as np
import os
size=4
haar_file = 'haarcascade_frontalface_default.xml'
dataset='dataset'

print('Recognizing Face Please Be in sufficient light')

(images,lables,names,id)=([],[],{},0)
for (subdirs,dirs,files) in os.walk(dataset):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(dataset,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath + '/' + filename
            lable=id
            images.append(cv2.imread(path,0))
            lables.append(int(lable))
        id=id+1

(width,height)=(130,100)

(images,lables)=[np.array(lis) for lis in [images,lables]]


model=cv2.face.LBPHFaceRecognizer_create()
model.train(images,lables)


face_cascade=cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)
while True:
    (_,im)=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        
        if prediction[1]<99:
            
            cv2.putText(im, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10),  cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
        else:
            cv2.putText(im,'not recognized',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
    
    cv2.imshow('OpenCV',im)
    
   
    if cv2.waitKey(1) & 0xFF==ord('q') :
        break
        
        