import cv2
import numpy as np
import face_recognition
import os
from djitellopy import Tello
from time import sleep



# me.streamon()
# # me.send_rc_control(0,0,25,0)
# time.sleep(2.2)



W, H = 360,240
fbRange = [5000, 7000]
pid = [0.4, 0.4, 0]
pError = 0
myFaceC = [0, 0]
myFaceArea = 0

path = 'C:/Users/Yashwanth/PycharmProjects/CVMajor/Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
## read images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


## function to find the major locations in the face
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding completed')

me = Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()


## Tracking Function
def trackFace( MyFaceC, MyFaceArea, W, pid, pError):

    area = MyFaceArea
    x,y = MyFaceC
    fb = 0

    error = x-W //2
    speed = pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed, -100, 100))



    if area>fbRange[0] and area<fbRange[1]: ## stable
        fb = 0
        # nanosense.sendData(['1'])
    if area> fbRange[1]: ## too close
        fb = -20
        # nanosense.sendData(['0'])
    elif area < fbRange[0] and area != 0: ## too far
        fb = 20
        # nanosense.sendData(['0'])

    if x == 0:
        speed = 0
        error = 0

    # print(speed, fb)
    me.send_rc_control(0,fb,0,speed)
    return error




# cap = cv2.VideoCapture(1)

## computing module : comparision, area and centre calculation etc is done
while True:
    # success, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img,(W,H))
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)


    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            # cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
            cx = (x1+x2) //2
            cy = (y1+y2) //2
            myFaceC = [cx,cy]
            w = abs(x1-x2)
            h = abs(y1-y2)
            area = w*h
            cv2.circle(img, (cx,cy), 5, (255,255,255),cv2.FILLED)
            # print(area)
            pError = trackFace(myFaceC, area, W, pid, pError)



    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break


