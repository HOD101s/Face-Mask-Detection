from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import cv2
    
with open('model.json', 'r') as f:
    loaded_model_json = f.read()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

resMap = {
        0 : 'Mask On',
        1 : 'Kindly Wear Mask'
    }

colorMap = {
        0 : (0,255,0),
        1 : (0,0,255)
    }

def prepImg(pth):
    return cv2.resize(pth,(224,224)).reshape(1,224,224,3)/255.0

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    faces = classifier.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,2)

    for face in faces:
        slicedImg = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
        pred = model.predict(prepImg(img))
        pred = np.argmax(pred)

        cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),colorMap[pred],2)
        cv2.putText(img, resMap[pred],(face[0],face[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)        
        
                
    cv2.imshow('FaceMask Detection',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
