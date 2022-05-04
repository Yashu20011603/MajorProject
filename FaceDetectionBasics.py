import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SerialModule import SerialObject

cap = cv2.VideoCapture(0)

detector = FaceDetector(minDetectionCon=0.8)

arduino = SerialObject(portNo='COM7')

while True:
    ss, img = cap.read()
    img, bbox = detector.findFaces(img)
    if bbox:
        arduino.sendData([1])
        # print('1')
    else:
        arduino.sendData([0])
        # print('0')p

    cv2.imshow("Image", img)
    cv2.waitKey(1)
