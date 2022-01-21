import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import pickle
from time import time, sleep
import RPi.GPIO as GPIO

# GPIO Setting
GPIO.setmode(GPIO.BCM)
SIG = 14
CNT = 0

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size = (640, 480))

def ultrasonic(GPIO_SIG):
    GPIO.setup(GPIO_SIG, GPIO.OUT)
    GPIO.output(GPIO_SIG, GPIO.LOW)
    sleep(0.2)
    GPIO.output(GPIO_SIG, GPIO.HIGH)
    sleep(0.5)
    GPIO.output(GPIO_SIG, GPIO.LOW)
    
    StartTime = time()
    StopTime = time()
    
    GPIO.setup(GPIO_SIG, GPIO.IN)
    while GPIO.input(GPIO_SIG) == 0:
        StartTime = time()
        
    while GPIO.input(GPIO_SIG) == 1:
        StopTime = time()
        
    Distance = (StopTime - StartTime) * 17150
    
    return Distance

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
lst = ["Minsu", "Unknown"]

while True:
    if ultrasonic(SIG) <= 50:
        break
    else:
        print("measured distance = %.1f cm" % ultrasonic(SIG))
        sleep(0.5)
        
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
    frame = frame.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 3)
    for (x,y,w,h) in faces:
        roiGray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roiGray)

        if conf > 70:
            if CNT == 10:
                print("\n[System] Registered person. Accepted!")
                print("\n[System] Door Unlocked!\n")
            name = lst[id_]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name + str(conf), (x,y), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            CNT += 1
            
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x,y), font, 2, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == 27 or CNT == 12:
        break

cv2.destroyAllWindows()
