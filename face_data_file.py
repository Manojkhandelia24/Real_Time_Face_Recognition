import cv2
import pandas as pd
import numpy as np
from time import sleep
import face_recognition as fr

fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)
name = input("Enter your name: ")
frameCount = 0
frameLimit = 20
names = []
enc = []

while True:
    flag, img = vid.read()
    if(flag):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                      
        faces = fd.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (50,50))  

        if len(faces) == 1:
            x,y,w,h = faces[0]
            img_face = img[y:y+h,x:x+w,:].copy()    ## Crop Image
            img_face = cv2.resize(img_face, (400,400), cv2.INTER_CUBIC)  ## Resize
            face_encoding = fr.face_encodings(img_face)  ## numpy array
            
            if len(face_encoding) == 1:
                enc.append(face_encoding[0].tolist())   # convert numpy array to list
                names.append(name)
                frameCount += 1

                cv2.putText(img, str(frameCount),(30,30), cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),5)

                if frameCount == frameLimit:
                    try:
                        old_data = pd.read_csv('face_data.csv',index_col = 0,sep='|')
                    except Exception as e:
                        print(e)
                    else:
                        enc_old = old_data['encoding'].values.tolist()
                        names_old = old_data['names'].values.tolist()
                        enc = enc_old + enc
                        names = names_old + names
                    data = {'names' : names, 'encoding' : enc}
                    pd.DataFrame(data).to_csv('face_data.csv', sep='|')   # encoding is in string format due to csv
                    break
            

        for x,y,w,h in faces:
            cv2.rectangle(img, pt1 = (x,y), pt2 = (x+w, y+h), color = (0,0,255), thickness = 8)
       
        cv2.imshow('Preview', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


cv2.destroyAllWindows()
vid.release()

face_data = pd.read_csv('face_data.csv',index_col = 0, sep = '|')
face_data['encoding'][0]   # string
eval(face_data['encoding'][0])
np.array(eval(face_data['encoding'][0]))
print(face_data)