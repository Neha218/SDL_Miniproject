
import cv2, os

#numpy for matrix calculation
import numpy as np

#Python Image Library (PIL)..Now known as PILLOW library
from PIL import Image


def assure_path_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
#LBPH(Local Binary Patterns Histograms) is the recognizer from opencv
recognizer = cv2.face.LBPHFaceRecognizer_create()

#prebuilt frontal face training model, for face detection, which we download from opencv
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Create method to get the images and label data
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    faceSamples=[]
    
    ids = []

    # Accessing image path from dataset one by one
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        #converting PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Splits the directory path to get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels('dataset')

# Training
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')
