import cv2
import imagezmq

image_hub = imagezmq.ImageHub()

while True:
  rpi_name, image = image_hub.recv_image()
  
  cv2.imshow(rpi_name, image)
  if cv2.waitKey(1) == ord('q'):
    cv2.imwrite("./capture_image/image.jpg", image)
    print("\n [System] Finished Capture!\n")
    break
  
  image_hub.send_reply(b'OK')


