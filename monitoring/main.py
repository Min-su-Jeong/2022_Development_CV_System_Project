import Adafruit_DHT
import time
import cv2

# Sensor setting
sensor = Adafruit_DHT.DHT11
pin = 15

# Cam setting
CAM_ID = 0
prev_time = 0
FPS = 10

cam = cv2.VideoCapture(CAM_ID) #카메라 생성
cam.set(3, 640)
cam.set(4, 480)
if cam.isOpened() == False: #카메라 생성 확인
    print('Can\'t open the CAM(%d)' % (CAM_ID))
    exit()

# Cam start
while True:
    #카메라에서 이미지 얻기
    ret, frame = cam.read()
    
    # Text 추가
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
    t = "temperature(C): "+ str(temperature)
    h = "Humidity(%): " + str(70)
    f = "Frame(fps): " + str(FPS)
    if temperature >= 38:
        warnt = t + "(Warning!!!)"
        cv2.putText(frame, warnt, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 2)
    else:
        cv2.putText(frame, t, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness = 2)
    if humidity >= 70:
        warnh = h + "(Warning!!!)"
        cv2.putText(frame, warnh, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 2)
    else:
        cv2.putText(frame, h, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness = 2)
    cv2.putText(frame, f, (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness = 2)
    
    current_time = time.time() - prev_time
    if (ret is True) and (current_time > 1./ FPS):
        prev_time = time.time()
        #얻어온 이미지 윈도우에 표시
        cv2.imshow('SmartFarm_Monitoring', frame)
    
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

#윈도우 종료
cam.release()
cv2.destroyWindow()