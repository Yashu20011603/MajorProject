import cv2
import cvzone

thres = 0.5
nmsThres = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):

            if classNames[classId -1] == 'dog':
                cv2.putText(img, f'{classNames[classId-1].upper()} {round(conf*100,2)}',
                            (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,
                            (0,255,0),2)
                cvzone.cornerRect(img, box, rt=0)
                cv2.circle(img, (box[0],box[1]),5,(255,0,0),cv2.FILLED)
                w = box[2]
                h = box[3]
                cx = box[0] + w // 2
                cy = box[1] + h // 2
                area = w * h
                print(area)


    except:
        pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)
