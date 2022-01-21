import cv2
import os
import numpy as np
from PIL import Image

path = "./face_capture/"
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Input face_id
face_id = input("\n enter face ID : ")
print("\n Intializing face capture. Look the Camera and wait...")

count = 0

while (True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the 'face_capture' folder
        cv2.imwrite(path + "User." + str(face_id) + '.' + str(count) + ".jpg", gray[y: y+h, x:x+w])
        cv2.imshow('image', img)
        print(count)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 100:
        break

print("\n Capture finished!\n")
cam.release()
cv2.destroyAllWindows()

path = './face_capture/'
recognizer = cv2.face.LBPHFaceRecognizer_create() # pip install opencv-contrib-python
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Get image & label data
def getData(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n[System] Training faces. It will take a few seconds. wait...")

faces, ids = getData(path)
recognizer.train(faces, np.array(ids))

# Save the model
recognizer.write('trainer.yml')

print("\n[System] {0} faces trained. Finished!\n".format(len(np.unique(ids))))
