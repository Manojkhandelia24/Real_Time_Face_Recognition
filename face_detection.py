import cv2
import numpy as np

from time import sleep

fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while True:    
    flag, img = vid.read()
    if flag:

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                      
        faces = fd.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (50,50))     # works better on gray image


        np.random.seed(50)
        colors = np.random.randint(0,255,(len(faces),3)).tolist()
        i = 0 

        for x,y,w,h in faces:
            cv2.rectangle(img, pt1 = (x,y), pt2 = (x+w, y+h), color = colors[i], thickness = 8)
            i+=1

        cv2.imshow('Preview',img)
        key = cv2.waitKey(1)
        if key == ord('q'):  # ord() gives the ascii value
            break
    else:
        print('No Frames')
        break
    sleep(0.1)

cv2.destroyAllWindows()
vid.release()
del vid