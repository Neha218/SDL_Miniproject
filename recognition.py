
#OpenCV2 for image processing
import cv2

# numpy for matrices calculations
import numpy as np

import os 

def assure_path_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#LBPH(Local Binary Patterns Histograms) is the recognizer from opencv

recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

#prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Creating classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

#font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

while True:
    # returns first a boolean indicating whether the read was successful, and then the image itself
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        #rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist 
        if(Id == 1):
            Id = "Shubham {0:.2f}%".format(round(100 - confidence, 2))

        # Put text which describes who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
