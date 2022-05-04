# Creating database 
# It captures images and stores them in datasets 
# folder under the folder name of sub_data 
import cv2, os 
import csv
import shutil
haar_file = 'haarcascade_frontalface_default.xml'


Id=(input("Enter ID"))
name=input("Enter Name")

# These are sub data sets of folder, 
# for my faces I've used my name you can 
# change the label here 
path = os.path.join('dataset',name) 
if not os.path.isdir(name): 
    #os.chdir('users\vihari\Documents\GitHub\Face-recognition\dataset')
    os.mkdir(name) 
    shutil.move(name,'dataset') 
path = os.path.join('dataset',name) 


# defining the size of images 
(width, height) = (130, 100)	 

#'0' is used for my webcam, 
# if you've any other camera 
# attached use '1' like this 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 

# The program loops until it has 30 images of the face. 
count = 1
while True: 
	(_, im) = webcam.read() 
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		count=count+1
		cv2.imwrite(path+"\\" +name +"."+Id +'.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
	    
	
	cv2.imshow('OpenCV', im) 
	  
	if cv2.waitKey(1) & 0xFF==ord('q') : 
		break
   
    
webcam.release()
cv2.destroyAllWindows() 
res = "Images Saved for ID : " + Id +" Name : "+ name
row = [Id , name]
with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()
4
